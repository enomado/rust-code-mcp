#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchPlan {
    pub start: usize,
    pub end: usize,
}

pub fn plan_batches<T>(
    items: &[T],
    max_batch_size: usize,
    max_tokens_per_batch: usize,
    mut token_len: impl FnMut(&T) -> usize,
) -> Vec<BatchPlan> {
    if items.is_empty() {
        return Vec::new();
    }

    let max_batch_size = max_batch_size.max(1);
    let max_tokens_per_batch = max_tokens_per_batch.max(1);
    let mut plans = Vec::new();
    let mut start = 0usize;
    let mut batch_len = 0usize;
    let mut batch_max_tokens = 0usize;

    for (idx, item) in items.iter().enumerate() {
        let item_tokens = token_len(item).max(1);
        let next_len = batch_len + 1;
        let next_max_tokens = batch_max_tokens.max(item_tokens);
        let exceeds_count = next_len > max_batch_size;
        let exceeds_token_budget = next_len * next_max_tokens > max_tokens_per_batch;

        if batch_len > 0 && (exceeds_count || exceeds_token_budget) {
            plans.push(BatchPlan { start, end: idx });
            start = idx;
            batch_len = 0;
            batch_max_tokens = 0;
        }

        batch_len += 1;
        batch_max_tokens = batch_max_tokens.max(item_tokens);
    }

    if batch_len > 0 {
        plans.push(BatchPlan {
            start,
            end: items.len(),
        });
    }

    plans
}

/// Высота батча — число СТРОК входа модели.
///
/// Ньютайп, а не `usize`, потому что рядом живут ещё две «величины батча»,
/// перепутать которые компилятор иначе не мешает: бюджет ТОКЕНОВ на батч
/// (`max_tokens_per_batch`) и число ЧАНКОВ, пришедших на индексацию. Все три
/// приезжают в один и тот же планировщик, и все три — `usize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BatchRows(pub usize);

/// Постоянная форма входа модели: `rows × seq_len`.
///
/// Существует только у рантаймов, которые компилируют ядра ПОД ФОРМУ
/// (сегодня — MIGraphX). У остальных формы нет вовсе: там паддинг до самой
/// длинной строки в батче бесплатен, и фиксировать нечего.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedInputShape {
    pub rows: BatchRows,
    pub seq_len: usize,
}

/// Кто диктует форму батча.
///
/// # Зачем это отдельный тип
/// Резать входы на батчи умеют ДВОЕ: индексатор (по своему `gpu_batch_size` и
/// токенному бюджету) и сам fastembed (по постоянной форме, если она задана).
/// Пока обе стороны режут молча, совпадение их дефолтов (оба 32) выглядит как
/// согласованность, а на деле держится ни на чём: подняв `gpu_batch_size` до
/// 100, получаешь два реза подряд — индексатор отдаёт по 100 строк, модель
/// режет их на 32+32+32+4 и добивает хвост до 32. Хвост появляется в КАЖДОМ
/// батче индексатора, а не один на весь прогон.
///
/// Поэтому решение «кто режет» принимается ОДИН раз и явно.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchingPolicy {
    /// Форму диктует МОДЕЛЬ: ровно `rows` строк в батче, хвост добивается ею же.
    ///
    /// Токенный бюджет и сортировка входов по длине здесь — мёртвая работа:
    /// каждая строка всё равно паддится до `seq_len`, поэтому соседство длинных
    /// и коротких на цену прогона не влияет.
    FixedByModel(BatchRows),
    /// Форму диктует ИНДЕКСАТОР: высота плюс потолок «строк × самая длинная
    /// строка», чтобы паддинг до длиннейшей не съедал память.
    ByBudget {
        max_rows: usize,
        max_tokens_per_batch: usize,
    },
}

impl BatchingPolicy {
    /// `fixed_rows` — форма, которую требует рантайм модели (`None`, если он
    /// формы не требует). Она ПЕРЕБИВАЕТ настройки индексатора: настройка,
    /// которая породила бы второй рез, не «компромисс», а лишняя работа.
    pub fn new(
        fixed_rows: Option<BatchRows>,
        max_rows: usize,
        max_tokens_per_batch: usize,
    ) -> Self {
        match fixed_rows {
            Some(rows) => Self::FixedByModel(rows),
            None => Self::ByBudget {
                max_rows: max_rows.max(1),
                max_tokens_per_batch: max_tokens_per_batch.max(1),
            },
        }
    }

    /// Число строк в полном батче — то, подо что реально идёт прогон модели.
    pub fn rows_per_batch(&self) -> BatchRows {
        match *self {
            Self::FixedByModel(rows) => rows,
            Self::ByBudget { max_rows, .. } => BatchRows(max_rows),
        }
    }

    /// Имеет ли смысл держать похожие по длине входы рядом.
    ///
    /// При постоянной форме — нет: длина каждой строки всё равно доводится до
    /// `seq_len`, и сортировка только тратит такты на входе в несколько тысяч
    /// чанков.
    pub fn sorts_by_length(&self) -> bool {
        matches!(self, Self::ByBudget { .. })
    }

    pub fn plan<T>(&self, items: &[T], token_len: impl FnMut(&T) -> usize) -> Vec<BatchPlan> {
        match *self {
            Self::FixedByModel(BatchRows(rows)) => plan_fixed_rows(items.len(), rows),
            Self::ByBudget {
                max_rows,
                max_tokens_per_batch,
            } => plan_batches(items, max_rows, max_tokens_per_batch, token_len),
        }
    }
}

impl std::fmt::Display for BatchingPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::FixedByModel(BatchRows(rows)) => write!(f, "model-fixed(rows={rows})"),
            Self::ByBudget {
                max_rows,
                max_tokens_per_batch,
            } => write!(f, "budget(rows={max_rows},tokens={max_tokens_per_batch})"),
        }
    }
}

/// Рез ровно по высоте формы: все батчи полные, кроме последнего.
///
/// Это и есть весь смысл политики `FixedByModel` — добивка случается один раз
/// на прогон, а не один раз на каждый батч индексатора.
fn plan_fixed_rows(len: usize, rows: usize) -> Vec<BatchPlan> {
    let rows = rows.max(1);
    (0..len)
        .step_by(rows)
        .map(|start| BatchPlan {
            start,
            end: (start + rows).min(len),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUDGET: usize = 49_152;

    /// Токенная длина не участвует в резе при постоянной форме — здесь она
    /// намеренно РАЗНАЯ, чтобы бюджет успел бы сработать, будь он в игре.
    fn items(n: usize) -> Vec<usize> {
        (0..n).map(|i| 8 + (i % 17) * 64).collect()
    }

    fn plan(policy: BatchingPolicy, n: usize) -> Vec<BatchPlan> {
        policy.plan(&items(n), |len| *len)
    }

    /// 🔑 Гейт DoD: при постоянной форме настройка индексатора на рез НЕ ВЛИЯЕТ.
    ///
    /// Мутант, на котором тест обязан краснеть: заставить `new` уважать
    /// `max_rows` при `Some(rows)` (то есть вернуть два реза подряд).
    #[test]
    fn fixed_shape_ignores_indexer_batch_size() {
        let rows = BatchRows(32);
        let reference = plan(BatchingPolicy::new(Some(rows), 32, BUDGET), 1000);
        for configured_rows in [1usize, 7, 31, 32, 64, 100, 1000] {
            for configured_tokens in [1usize, 4096, BUDGET] {
                let policy = BatchingPolicy::new(Some(rows), configured_rows, configured_tokens);
                assert_eq!(
                    plan(policy, 1000),
                    reference,
                    "настройки индексатора ({configured_rows}, {configured_tokens}) \
                     изменили рез при постоянной форме модели"
                );
            }
        }
    }

    /// Добивается РОВНО один батч на прогон, а не по одному на каждый рез
    /// индексатора. Это и есть цена, ради которой п.8 бэклога заводился.
    #[test]
    fn fixed_shape_pads_only_the_tail() {
        let rows = 32usize;
        for n in [1usize, 31, 32, 33, 64, 100, 999, 1000] {
            let plans = plan(BatchingPolicy::new(Some(BatchRows(rows)), 100, BUDGET), n);
            let padded_rows: usize = plans.iter().map(|p| rows - (p.end - p.start)).sum();
            assert_eq!(
                padded_rows,
                rows * n.div_ceil(rows) - n,
                "добивка сверх одного хвоста при n={n}"
            );
            assert!(
                plans
                    .iter()
                    .take(plans.len().saturating_sub(1))
                    .all(|p| p.end - p.start == rows),
                "неполный батч не последний при n={n}"
            );
        }
    }

    /// Покрытие: рез обязан покрыть вход целиком и без пересечений — иначе
    /// «эмбеддингов меньше, чем чанков» вылезет уже в индексе.
    #[test]
    fn fixed_shape_covers_every_input_exactly_once() {
        for n in [0usize, 1, 5, 32, 33, 257] {
            let plans = plan(BatchingPolicy::new(Some(BatchRows(32)), 32, BUDGET), n);
            let covered: Vec<usize> = plans.iter().flat_map(|p| p.start..p.end).collect();
            assert_eq!(
                covered,
                (0..n).collect::<Vec<_>>(),
                "рез не покрыл вход при n={n}"
            );
        }
    }

    /// Без постоянной формы политика остаётся прежней: и высота, и токенный
    /// бюджет действуют. Позитивный контроль — без него тесты выше зелены и на
    /// коде, который просто игнорирует настройки ВСЕГДА.
    #[test]
    fn budget_policy_still_honours_both_limits() {
        let policy = BatchingPolicy::new(None, 8, BUDGET);
        assert!(policy.sorts_by_length());
        let plans = plan(policy, 20);
        assert!(plans.iter().all(|p| p.end - p.start <= 8));
        assert_eq!(plans.iter().map(|p| p.end - p.start).sum::<usize>(), 20);

        // Токенный бюджет режет РАНЬШЕ высоты: 4 строки по 1000 токенов
        // не помещаются в бюджет 2048.
        let long = vec![1000usize; 4];
        let plans = BatchingPolicy::new(None, 8, 2048).plan(&long, |len| *len);
        assert_eq!(plans.len(), 2, "токенный бюджет перестал резать");
    }

    #[test]
    fn fixed_policy_does_not_sort_and_reports_its_rows() {
        let policy = BatchingPolicy::new(Some(BatchRows(32)), 100, BUDGET);
        assert!(!policy.sorts_by_length());
        assert_eq!(policy.rows_per_batch(), BatchRows(32));
        assert_eq!(policy.to_string(), "model-fixed(rows=32)");
    }
}
