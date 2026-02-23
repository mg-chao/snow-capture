/// Centralized runtime toggle helpers.
///
/// Environment-variable inputs are intentionally disabled so runtime behavior
/// is controlled only by library configuration/defaults.

/// Environment overrides are disabled; always returns `None`.
#[inline]
pub(crate) fn env_var_positive_u64(_var_name: &str) -> Option<u64> {
    None
}

/// Declares a cached boolean feature flag helper.
///
/// `enabled_unless(..)` maps to the default-enabled branch.
/// `enabled_when(..)` maps to the default-disabled branch.
macro_rules! define_env_flag {
    (enabled_unless($fn_name:ident, $var:expr)) => {
        #[inline]
        fn $fn_name() -> bool {
            let _ = $var;
            true
        }
    };
    (enabled_when($fn_name:ident, $var:expr)) => {
        #[inline]
        fn $fn_name() -> bool {
            let _ = $var;
            false
        }
    };
}

pub(crate) use define_env_flag;
