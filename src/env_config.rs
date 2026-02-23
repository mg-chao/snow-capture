/// Centralized environment-variable parsing helpers.
///
/// All feature-toggle and tuning-knob env-var reads should go through these
/// helpers so the truthy/falsey parsing logic lives in exactly one place.

/// Returns `true` when the environment variable is set to a truthy value
/// (`1`, `true`, `yes`, or `on`, case-insensitive, trimmed).
#[inline]
pub(crate) fn env_var_truthy(var_name: &str) -> bool {
    std::env::var(var_name)
        .map(|raw| {
            let normalized = raw.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on"
        })
        .unwrap_or(false)
}

/// Parses the environment variable as a `u64`, returning `Some` only when
/// the value is a valid positive (> 0) integer.
#[inline]
pub(crate) fn env_var_positive_u64(var_name: &str) -> Option<u64> {
    std::env::var(var_name)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
}

/// Declares a cached boolean feature flag backed by a `OnceLock<bool>`.
///
/// # Variants
///
/// `enabled_unless(fn_name, "ENV_VAR")` — returns `true` unless the env var
/// is truthy (i.e. the feature is on by default, disabled by the env var).
///
/// `enabled_when(fn_name, "ENV_VAR")` — returns `true` only when the env var
/// is truthy (i.e. the feature is off by default, enabled by the env var).
macro_rules! define_env_flag {
    (enabled_unless($fn_name:ident, $var:expr)) => {
        #[inline]
        fn $fn_name() -> bool {
            static VALUE: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
            *VALUE.get_or_init(|| !$crate::env_config::env_var_truthy($var))
        }
    };
    (enabled_when($fn_name:ident, $var:expr)) => {
        #[inline]
        fn $fn_name() -> bool {
            static VALUE: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
            *VALUE.get_or_init(|| $crate::env_config::env_var_truthy($var))
        }
    };
}

pub(crate) use define_env_flag;
