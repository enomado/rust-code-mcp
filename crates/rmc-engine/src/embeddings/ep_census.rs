//! Оракул «граф РЕАЛЬНО посчитан на execution provider'е».
//!
//! # Зачем отдельный слой
//! `error_on_failure()` на EP ловит РОВНО один отказ: «провайдер не
//! зарегистрировался». Случай «EP поднялся, но ни один узел ему не достался, и
//! граф целиком посчитан на CPU» проходит эту проверку насквозь: сессия
//! создаётся, числа считаются, ошибок нет. Отличить его от успеха можно было
//! только косвенно — по скорости, то есть глазом и не в тесте.
//!
//! Единственный известный прямой ответ даёт профиль ONNX Runtime: в нём у
//! КАЖДОГО события узла стоит имя провайдера, который этот узел исполнил.
//! Отсюда и весь модуль — превратить профиль в перепись «узлов по
//! провайдерам», по которой уже можно утверждать (и ассертить), где считалось.
//!
//! Формат событий ORT: массив объектов, у узловых `"cat": "Node"`, имя
//! провайдера — в `args.provider`. На один узел приходится НЕСКОЛЬКО событий
//! (`_fence_before`, `_kernel_time`, `_fence_after`) и по одному комплекту на
//! КАЖДЫЙ прогон, поэтому перепись ведётся по ИМЕНАМ узлов, а не по событиям:
//! иначе число зависело бы от того, сколько раз погоняли модель, и «4158» ни о
//! чём бы не говорило.

use crate::embeddings::EmbeddingError;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

/// Имя MIGraphX EP в профиле ORT — то, чем провайдер подписывает свои узлы.
pub const MIGRAPHX_EP: &str = "MIGraphXExecutionProvider";
/// Имя CPU EP в профиле ORT. Узлы формы (Shape/Reshape/Cast) остаются на нём
/// даже при полностью здоровом GPU-пути — их горстка, и это норма.
pub const CPU_EP: &str = "CPUExecutionProvider";

/// Сколько РАЗНЫХ узлов графа исполнил каждый провайдер.
///
/// Инвариант: перепись непуста. Пустая (профиль без provider-меток) — это
/// «не смотрели», а не «ничего не на GPU», и она не должна быть неотличима от
/// честного нуля, поэтому конструкторы в этом случае возвращают ошибку.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderCensus {
    per_provider: BTreeMap<String, usize>,
}

impl ProviderCensus {
    /// Перепись из файла, который вернул `end_profiling()`.
    pub fn from_profile_file(path: &Path) -> Result<Self, EmbeddingError> {
        let raw = std::fs::read_to_string(path).map_err(|e| {
            EmbeddingError::model_init(format!(
                "cannot read ORT profile at {}: {e}",
                path.display()
            ))
        })?;
        Self::from_profile_json(&raw)
    }

    /// Перепись из содержимого профиля.
    ///
    /// Отказы вместо тихой подгонки: не-массив и профиль без единой
    /// provider-метки — ошибки. Второе особенно важно: молча вернуть пустую
    /// перепись значит отдать вызывающему ноль, который он прочтёт как
    /// «всё посчиталось на CPU», хотя на деле профиль просто не о том.
    pub fn from_profile_json(raw: &str) -> Result<Self, EmbeddingError> {
        let events: serde_json::Value = serde_json::from_str(raw)
            .map_err(|e| EmbeddingError::model_init(format!("malformed ORT profile: {e}")))?;
        let events = events.as_array().ok_or_else(|| {
            EmbeddingError::model_init("ORT profile is not a JSON array of events")
        })?;

        // Имя узла → провайдер. Множество, а не счётчик, потому что события
        // одного узла повторяются на каждом прогоне.
        let mut seen: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for event in events {
            if event.get("cat").and_then(|c| c.as_str()) != Some("Node") {
                continue;
            }
            let Some(provider) = event
                .get("args")
                .and_then(|a| a.get("provider"))
                .and_then(|p| p.as_str())
            else {
                continue;
            };
            let Some(name) = event.get("name").and_then(|n| n.as_str()) else {
                continue;
            };
            seen.entry(node_name(name).to_string())
                .or_default()
                .insert(provider.to_string());
        }

        if seen.is_empty() {
            return Err(EmbeddingError::model_init(
                "ORT profile carries no provider-tagged node events — \
                 profiling was probably not enabled for this session",
            ));
        }

        let mut per_provider: BTreeMap<String, usize> = BTreeMap::new();
        for providers in seen.values() {
            // Один узел исполняется одним провайдером; если профиль утверждает
            // иное, считаем узел за каждого — молча выбирать «первого» значило
            // бы прятать то, чего мы не понимаем.
            for provider in providers {
                *per_provider.entry(provider.clone()).or_insert(0) += 1;
            }
        }
        Ok(Self { per_provider })
    }

    pub fn per_provider(&self) -> &BTreeMap<String, usize> {
        &self.per_provider
    }

    pub fn nodes_on(&self, provider: &str) -> usize {
        self.per_provider.get(provider).copied().unwrap_or(0)
    }

    pub fn total_nodes(&self) -> usize {
        self.per_provider.values().sum()
    }

    /// Доля узлов, доставшихся провайдеру, в [0, 1].
    ///
    /// Знаменатель — всегда непустой (см. инвариант типа), деления на ноль тут
    /// не бывает по построению.
    pub fn share_on(&self, provider: &str) -> f64 {
        self.nodes_on(provider) as f64 / self.total_nodes() as f64
    }
}

impl fmt::Display for ProviderCensus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .per_provider
            .iter()
            .map(|(provider, count)| format!("{provider}={count}"))
            .collect();
        write!(f, "{} nodes: {}", self.total_nodes(), parts.join(", "))
    }
}

/// Имя узла без суффикса, которым ORT помечает ФАЗУ события.
///
/// `Add_12_kernel_time`, `Add_12_fence_before` и `Add_12_fence_after` — один и
/// тот же узел графа; без срезки суффикса он попал бы в перепись трижды.
fn node_name(event_name: &str) -> &str {
    for suffix in ["_kernel_time", "_fence_before", "_fence_after"] {
        if let Some(stripped) = event_name.strip_suffix(suffix) {
            return stripped;
        }
    }
    event_name
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Профиль в форме, которую пишет ORT: у узла три события, прогонов два.
    fn profile(nodes: &[(&str, &str)], runs: usize) -> String {
        let mut events = vec![serde_json::json!({
            "cat": "Session", "name": "model_loading_uri", "dur": 1
        })];
        for _ in 0..runs {
            for (name, provider) in nodes {
                for phase in ["_fence_before", "_kernel_time", "_fence_after"] {
                    events.push(serde_json::json!({
                        "cat": "Node",
                        "name": format!("{name}{phase}"),
                        "dur": 7,
                        "args": {"provider": provider, "op_name": "Add"},
                    }));
                }
            }
        }
        serde_json::Value::Array(events).to_string()
    }

    /// Перепись считает УЗЛЫ, а не события: три события на узел и два прогона
    /// не превращают два узла в двенадцать.
    ///
    /// Это главное свойство: если бы число ехало от количества прогонов, любой
    /// порог на долю GPU-узлов держался бы на том, сколько раз погоняли модель.
    #[test]
    fn census_counts_nodes_not_events() {
        let raw = profile(&[("Add_1", MIGRAPHX_EP), ("Shape_2", CPU_EP)], 2);
        let census = ProviderCensus::from_profile_json(&raw).unwrap();
        assert_eq!(census.nodes_on(MIGRAPHX_EP), 1);
        assert_eq!(census.nodes_on(CPU_EP), 1);
        assert_eq!(census.total_nodes(), 2);
        assert!((census.share_on(MIGRAPHX_EP) - 0.5).abs() < 1e-9);
    }

    /// Тихий откат на CPU виден переписью — ровно тот класс, ради которого
    /// модуль и заведён.
    #[test]
    fn census_sees_silent_cpu_fallback() {
        let raw = profile(&[("Add_1", CPU_EP), ("MatMul_2", CPU_EP)], 1);
        let census = ProviderCensus::from_profile_json(&raw).unwrap();
        assert_eq!(census.nodes_on(MIGRAPHX_EP), 0);
        assert_eq!(census.share_on(CPU_EP), 1.0);
    }

    /// «Не смотрели» ≠ «ничего не на GPU»: профиль без provider-меток обязан
    /// быть отказом, иначе ноль GPU-узлов не отличить от отсутствия данных.
    #[test]
    fn census_rejects_profile_without_provider_tags() {
        let raw = serde_json::json!([
            {"cat": "Session", "name": "session_initialization", "dur": 1},
            {"cat": "Node", "name": "Add_1_kernel_time", "dur": 2, "args": {"op_name": "Add"}},
        ])
        .to_string();
        let err = ProviderCensus::from_profile_json(&raw).unwrap_err();
        assert!(
            err.to_string().contains("no provider-tagged node events"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn census_rejects_malformed_profile() {
        assert!(ProviderCensus::from_profile_json("{}").is_err());
        assert!(ProviderCensus::from_profile_json("not json").is_err());
    }
}
