// Политику собирают ВСЕГДА, а зовут её только из пути под фичей
// `embeddings-migraphx`. Гейты (чистая функция вытеснения, разбор потолка,
// сцена на настоящих каталогах) обязаны гоняться в обычной суите — на машине
// без AMD-карты и без системного ORT их иначе не запустит никто.
#![cfg_attr(not(feature = "embeddings-migraphx"), allow(dead_code))]

//! Политика кэша скомпилированных MIGraphX-ядер.
//!
//! # Что тут вообще хранится
//! MIGraphX компилирует ядра ПОД ФОРМУ входа и складывает результат в `.mxr`:
//! одна форма — 45–70 с компиляции и ~150–200 МБ на диске. Каталог адресуется
//! моделью и формой (`<модель>-<rows>x<seq_len>`), поэтому программы разных
//! форм физически не встречаются (см. `ensure_migraphx_kernel_cache`).
//!
//! # Почему нужна политика, а не только адресация
//! Адресация формой делает рост ПРЕДСКАЗУЕМЫМ, но не ОГРАНИЧЕННЫМ: каталог
//! остаётся навсегда, даже когда его форма больше недостижима — сменили
//! `max_len` в профиле, подняли высоту батча, попробовали вторую модель. Каждый
//! такой шаг оставляет ~200 МБ, которые никто уже не прочитает, и никто их не
//! убирает. Отсюда потолок с вытеснением.
//!
//! # Почему потолок, а не «снести чужие формы»
//! «Оставить только текущую форму» выглядит точнее, но ломается ровно там, где
//! кэш и нужен: две конфигурации на одной машине (например, профиль с другим
//! `max_len`, или второй сервер) начали бы сносить кэш друг друга по кругу,
//! оплачивая по 45–70 с компиляции на каждом старте. Потолок такого цикла не
//! даёт: пока сумма влезает, живут обе формы.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Имя метки «этой формой пользовались».
///
/// Файл трогается на КАЖДОЙ инициализации эмбеддера, а `.mxr` пишется только
/// при компиляции. Без метки «давность» означала бы «когда скомпилировали», и
/// форма, которой пользуются каждый день, вытеснялась бы раньше формы,
/// скомпилированной вчера и заброшенной.
pub(crate) const LAST_USED_MARKER: &str = ".last-used";

/// Каталог одной формы в кэше ядер.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CachedShape {
    pub path: PathBuf,
    pub bytes: u64,
    pub last_used: SystemTime,
}

/// Что снести и что останется после.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvictionPlan {
    /// Каталоги под снос, в порядке вытеснения (самые давние первыми).
    pub remove: Vec<PathBuf>,
    /// Сколько байт останется, если план выполнить.
    pub bytes_after: u64,
    /// Потолок недостижим даже после сноса ВСЕГО остального — то есть его
    /// выставили ниже цены одной формы. Отдельный флаг, а не молчаливое
    /// «снесли всё, что смогли»: иначе кэш каждый раз оставался бы над
    /// потолком, и это выглядело бы как неработающая политика.
    pub still_over_cap: bool,
}

/// Спланировать вытеснение: снести самые давние формы, пока сумма не влезет
/// в потолок.
///
/// `keep` не сносится НИКОГДА — это форма, которую прямо сейчас поднимает
/// текущий процесс. Снести её означало бы гарантированную перекомпиляцию на
/// следующем же прогоне, то есть кэш, который сам себя обнуляет.
///
/// `cap_bytes == 0` — вытеснение отключено (явный отказ от политики, не
/// «потолок в ноль»).
pub(crate) fn plan_eviction(shapes: &[CachedShape], keep: &Path, cap_bytes: u64) -> EvictionPlan {
    let total: u64 = shapes.iter().map(|shape| shape.bytes).sum();
    if cap_bytes == 0 || total <= cap_bytes {
        return EvictionPlan {
            remove: Vec::new(),
            bytes_after: total,
            still_over_cap: false,
        };
    }

    let mut candidates: Vec<&CachedShape> =
        shapes.iter().filter(|shape| shape.path != keep).collect();
    // Давность — первый ключ, путь — второй: при равных отметках времени
    // (файловые системы с секундной гранулярностью) порядок обязан быть
    // ДЕТЕРМИНИРОВАННЫМ, иначе один и тот же кэш вытесняется по-разному от
    // прогона к прогону.
    candidates.sort_by(|a, b| {
        a.last_used
            .cmp(&b.last_used)
            .then_with(|| a.path.cmp(&b.path))
    });

    let mut remaining = total;
    let mut remove = Vec::new();
    for shape in candidates {
        if remaining <= cap_bytes {
            break;
        }
        remaining -= shape.bytes;
        remove.push(shape.path.clone());
    }

    EvictionPlan {
        remove,
        bytes_after: remaining,
        still_over_cap: remaining > cap_bytes,
    }
}

/// Суммарный размер файлов под каталогом.
///
/// Ошибки чтения отдельных элементов пропускаются намеренно: каталог кэша
/// может меняться под нами (соседний процесс компилирует свою форму), и
/// сорвать инициализацию эмбеддера из-за гонки в УБОРКЕ — хуже, чем посчитать
/// на несколько мегабайт мимо.
pub(crate) fn dir_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => dir_size(&entry.path()),
            Ok(_) => entry.metadata().map(|meta| meta.len()).unwrap_or(0),
            Err(_) => 0,
        })
        .sum()
}

/// Прочитать состояние кэша: по каталогу на форму.
///
/// Давность берётся из метки [`LAST_USED_MARKER`]; её отсутствие — не отказ, а
/// каталог от сборки, которая метку ещё не ставила. Для него давностью служит
/// собственное время каталога, то есть «когда компилировали» — единственная
/// отметка, которая там есть.
pub(crate) fn read_cached_shapes(root: &Path) -> Vec<CachedShape> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut shapes = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
            continue;
        }
        let last_used = std::fs::metadata(path.join(LAST_USED_MARKER))
            .or_else(|_| std::fs::metadata(&path))
            .and_then(|meta| meta.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        shapes.push(CachedShape {
            bytes: dir_size(&path),
            path,
            last_used,
        });
    }
    shapes
}

/// Отметить форму использованной прямо сейчас.
pub(crate) fn touch_last_used(dir: &Path) {
    // Пересоздание файла двигает mtime — этого и достаточно. Отказ здесь не
    // фатален: без метки форма просто уйдёт под правило «давность = время
    // каталога».
    let _ = std::fs::File::create(dir.join(LAST_USED_MARKER));
}

/// Привести кэш к потолку. Возвращает выполненный план.
pub(crate) fn sweep(root: &Path, keep: &Path, cap_bytes: u64) -> EvictionPlan {
    let shapes = read_cached_shapes(root);
    let plan = plan_eviction(&shapes, keep, cap_bytes);
    for path in &plan.remove {
        match std::fs::remove_dir_all(path) {
            Ok(()) => tracing::info!(
                target: "embeddings::kernel_cache",
                evicted = %path.display(),
                "evicted a MIGraphX kernel cache shape"
            ),
            Err(err) => tracing::warn!(
                target: "embeddings::kernel_cache",
                path = %path.display(),
                error = %err,
                "cannot evict a MIGraphX kernel cache shape"
            ),
        }
    }
    if plan.still_over_cap {
        tracing::warn!(
            target: "embeddings::kernel_cache",
            bytes_after = plan.bytes_after,
            "MIGraphX kernel cache is over its cap even after eviction: \
             the cap is below the size of a single shape"
        );
    }
    plan
}

/// Переменная, которой потолок кэша ядер переопределяют.
pub(crate) const CAP_ENV: &str = "RMC_MIGRAPHX_KERNEL_CACHE_MAX_BYTES";

/// Потолок по умолчанию — 1 ГиБ, то есть примерно пять форм по ~200 МБ.
///
/// Величина выбрана из ЗАМЕРЕННОЙ цены одной формы, а не из круглого числа:
/// меньше двух форм — и любая вторая конфигурация на машине начнёт платить
/// компиляцию; больше пяти — и кэш растёт быстрее, чем кто-либо заметит.
pub(crate) const DEFAULT_CAP_BYTES: u64 = 1 << 30;

/// Разобрать потолок: голое число байт либо число с суффиксом `K`/`M`/`G`
/// (степени 1024). `0` — отключить вытеснение.
///
/// Мусор — ОШИБКА, а не «возьмём дефолт»: молча проигнорированный потолок
/// читается как «я его выставил», и кэш растёт вопреки настройке.
pub(crate) fn parse_cap_bytes(raw: &str) -> Result<u64, String> {
    let raw = raw.trim();
    let (digits, multiplier) = match raw.chars().last() {
        Some('K') | Some('k') => (&raw[..raw.len() - 1], 1024),
        Some('M') | Some('m') => (&raw[..raw.len() - 1], 1024 * 1024),
        Some('G') | Some('g') => (&raw[..raw.len() - 1], 1024 * 1024 * 1024),
        _ => (raw, 1),
    };
    let value: u64 = digits
        .trim()
        .parse()
        .map_err(|_| format!("`{raw}` is not a byte count (expected e.g. `2G`, `512M`, `0`)"))?;
    value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("`{raw}` overflows a byte count"))
}

/// Потолок из окружения, либо [`DEFAULT_CAP_BYTES`].
pub(crate) fn cap_bytes_from_env() -> Result<u64, String> {
    match std::env::var(CAP_ENV) {
        Ok(raw) if !raw.trim().is_empty() => {
            parse_cap_bytes(&raw).map_err(|err| format!("{CAP_ENV}: {err}"))
        }
        _ => Ok(DEFAULT_CAP_BYTES),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn shape(name: &str, bytes: u64, age_secs: u64) -> CachedShape {
        CachedShape {
            path: PathBuf::from("/cache").join(name),
            bytes,
            last_used: SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000 - age_secs),
        }
    }

    const CAP: u64 = 500;

    #[test]
    fn under_the_cap_nothing_is_evicted() {
        let shapes = [shape("a", 200, 10), shape("b", 200, 20)];
        let plan = plan_eviction(&shapes, Path::new("/cache/a"), CAP);
        assert!(plan.remove.is_empty());
        assert_eq!(plan.bytes_after, 400);
        assert!(!plan.still_over_cap);
    }

    /// Вытесняются САМЫЕ ДАВНИЕ и ровно до потолка — не всё подряд.
    #[test]
    fn evicts_the_least_recently_used_until_it_fits() {
        let shapes = [
            shape("fresh", 200, 1),
            shape("stale", 200, 100),
            shape("ancient", 200, 999),
            shape("current", 200, 50),
        ];
        let plan = plan_eviction(&shapes, Path::new("/cache/current"), CAP);
        assert_eq!(
            plan.remove,
            vec![
                PathBuf::from("/cache/ancient"),
                PathBuf::from("/cache/stale"),
            ]
        );
        assert_eq!(plan.bytes_after, 400);
        assert!(!plan.still_over_cap);
    }

    /// 🔑 Текущая форма не сносится даже когда она самая давняя — иначе кэш
    /// обнулял бы сам себя и платил компиляцию на каждом прогоне.
    #[test]
    fn the_shape_in_use_is_never_evicted() {
        let shapes = [
            shape("current", 400, 999),
            shape("fresh_a", 200, 1),
            shape("fresh_b", 200, 2),
        ];
        let plan = plan_eviction(&shapes, Path::new("/cache/current"), CAP);
        assert!(
            !plan.remove.contains(&PathBuf::from("/cache/current")),
            "снесена форма, которую прямо сейчас поднимают"
        );
        assert_eq!(plan.remove.len(), 2);
        assert_eq!(plan.bytes_after, 400);
    }

    /// Потолок ниже одной формы — вытеснение не молчит, а называет исход.
    #[test]
    fn a_cap_below_one_shape_is_reported_not_hidden() {
        let shapes = [shape("current", 900, 5), shape("other", 200, 6)];
        let plan = plan_eviction(&shapes, Path::new("/cache/current"), CAP);
        assert_eq!(plan.remove, vec![PathBuf::from("/cache/other")]);
        assert_eq!(plan.bytes_after, 900);
        assert!(
            plan.still_over_cap,
            "кэш остался над потолком, а политика об этом промолчала"
        );
    }

    /// Ноль — явный отказ от политики, а не «потолок в ноль байт».
    #[test]
    fn zero_cap_disables_eviction() {
        let shapes = [shape("a", 10_000, 1), shape("b", 10_000, 2)];
        let plan = plan_eviction(&shapes, Path::new("/cache/a"), 0);
        assert!(plan.remove.is_empty());
        assert_eq!(plan.bytes_after, 20_000);
        assert!(!plan.still_over_cap);
    }

    /// Одинаковые отметки времени (секундная гранулярность ФС) не должны
    /// давать разный порядок вытеснения от прогона к прогону.
    #[test]
    fn equal_timestamps_evict_deterministically() {
        let shapes = [
            shape("z", 200, 42),
            shape("a", 200, 42),
            shape("m", 200, 42),
            shape("current", 200, 42),
        ];
        let first = plan_eviction(&shapes, Path::new("/cache/current"), CAP);
        let reversed: Vec<CachedShape> = shapes.iter().rev().cloned().collect();
        let second = plan_eviction(&reversed, Path::new("/cache/current"), CAP);
        assert_eq!(first, second);
        assert_eq!(
            first.remove,
            vec![PathBuf::from("/cache/a"), PathBuf::from("/cache/m")]
        );
    }

    /// Сквозная сцена на настоящих каталогах: метка решает, кто давнее, и
    /// снос действительно происходит.
    #[test]
    fn sweep_removes_directories_on_disk() {
        let root =
            std::env::temp_dir().join(format!("rmc-kernel-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let keep = root.join("keep");
        let stale = root.join("stale");
        for dir in [&keep, &stale] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("kernels.mxr"), vec![0u8; 300]).unwrap();
        }
        // `stale` помечен использованным раньше `keep`: обе метки ставим явно,
        // чтобы сцена не зависела от порядка создания каталогов.
        touch_last_used(&stale);
        std::thread::sleep(Duration::from_millis(1100));
        touch_last_used(&keep);

        let plan = sweep(&root, &keep, CAP);
        assert_eq!(plan.remove, vec![stale.clone()]);
        assert!(!stale.exists(), "каталог остался на диске");
        assert!(keep.exists(), "снесена форма, которая в работе");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cap_parser_takes_suffixes_and_rejects_garbage() {
        assert_eq!(parse_cap_bytes("0").unwrap(), 0);
        assert_eq!(parse_cap_bytes("1024").unwrap(), 1024);
        assert_eq!(parse_cap_bytes("512M").unwrap(), 512 * 1024 * 1024);
        assert_eq!(parse_cap_bytes(" 2G ").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_cap_bytes("1k").unwrap(), 1024);
        assert!(parse_cap_bytes("").is_err());
        assert!(parse_cap_bytes("много").is_err());
        assert!(parse_cap_bytes("2GB").is_err());
        assert!(parse_cap_bytes("-1").is_err());
        assert!(
            parse_cap_bytes("99999999999999999999G").is_err(),
            "переполнение прошло молча"
        );
    }
}
