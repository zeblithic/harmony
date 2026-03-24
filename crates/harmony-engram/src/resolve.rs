use alloc::vec::Vec;
use crate::{EngramConfig, EngramError, EngramLookup};

pub fn aggregate(
    _config: &EngramConfig,
    _lookup: &EngramLookup,
    _shard_data: &[&[u8]],
) -> Result<Vec<u8>, EngramError> {
    todo!()
}
