use thiserror::Error;

#[derive(Debug, Error)]
pub enum OwnerError {
    #[error("placeholder")]
    Placeholder,
}
