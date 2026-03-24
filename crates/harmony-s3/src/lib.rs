//! Async S3 client wrapper for storing and retrieving immutable books by
//! content identifier.
//!
//! # Design
//!
//! [`S3Library`] is a thin, typed wrapper around `aws-sdk-s3` that maps
//! 32-byte content IDs to S3 object keys and exposes only the three
//! operations needed for a content-addressed store: put, get, and exists.
//!
//! All objects are stored with `INTELLIGENT_TIERING` storage class so that
//! infrequently-accessed books automatically migrate to cheaper tiers without
//! any application-layer coordination.

mod error;

pub use error::S3Error;

use aws_config::BehaviorVersion;
use aws_sdk_s3::{
    primitives::ByteStream,
    types::StorageClass,
    Client,
};
use tracing::{debug, instrument};

/// Content-addressed S3 client.
///
/// Keys are formed as `{prefix}book/{hex_content_id}` where `hex_content_id`
/// is the lowercase hex encoding of a 32-byte content identifier.
#[derive(Debug, Clone)]
pub struct S3Library {
    client: Client,
    bucket: String,
    prefix: String,
}

impl S3Library {
    /// Create a new [`S3Library`] client.
    ///
    /// Configuration is loaded from the environment (AWS_REGION,
    /// AWS_ACCESS_KEY_ID, etc.) with `region` overriding the ambient region.
    ///
    /// # Errors
    ///
    /// Returns [`S3Error::ConfigError`] if the AWS SDK cannot be configured.
    #[instrument(skip_all)]
    pub async fn new(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        region: impl Into<String>,
    ) -> Result<Self, S3Error> {
        let bucket = bucket.into();
        let prefix = prefix.into();
        let region_str = region.into();

        let region = aws_config::Region::new(region_str);
        let config = aws_config::defaults(BehaviorVersion::latest())
            .region(region)
            .load()
            .await;
        let client = Client::new(&config);

        debug!("S3Library initialised");
        Ok(Self {
            client,
            bucket,
            prefix,
        })
    }

    /// Compute the S3 object key for a content identifier.
    ///
    /// Format: `{prefix}book/{hex_content_id}`
    pub fn object_key(&self, cid_bytes: &[u8; 32]) -> String {
        format!("{}book/{}", self.prefix, hex::encode(cid_bytes))
    }

    /// Store a book in S3.
    ///
    /// Uses `INTELLIGENT_TIERING` storage class. The call is idempotent —
    /// re-uploading the same content identifier overwrites with identical data.
    ///
    /// # Errors
    ///
    /// Returns [`S3Error::PutFailed`] if the upload fails.
    #[instrument(skip(self, data), fields(bucket = %self.bucket))]
    pub async fn put_book(&self, cid_bytes: &[u8; 32], data: Vec<u8>) -> Result<(), S3Error> {
        let key = self.object_key(cid_bytes);
        debug!(key, bytes = data.len(), "putting book");

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .storage_class(StorageClass::IntelligentTiering)
            .body(ByteStream::from(data))
            .send()
            .await
            .map_err(|e| S3Error::PutFailed(e.to_string()))?;

        Ok(())
    }

    /// Retrieve a book from S3.
    ///
    /// Returns `None` if the object does not exist (`NoSuchKey`), or `Some`
    /// with the raw bytes on success.
    ///
    /// # Errors
    ///
    /// Returns [`S3Error::GetFailed`] for any error other than `NoSuchKey`.
    #[instrument(skip(self), fields(bucket = %self.bucket))]
    pub async fn get_book(&self, cid_bytes: &[u8; 32]) -> Result<Option<Vec<u8>>, S3Error> {
        let key = self.object_key(cid_bytes);
        debug!(key, "getting book");

        let result = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await;

        match result {
            Ok(output) => {
                let bytes = output
                    .body
                    .collect()
                    .await
                    .map_err(|e| S3Error::GetFailed(e.to_string()))?
                    .into_bytes()
                    .to_vec();
                Ok(Some(bytes))
            }
            Err(sdk_err) => {
                let svc_err = sdk_err.into_service_error();
                if svc_err.is_no_such_key() {
                    debug!(key, "book not found");
                    Ok(None)
                } else {
                    Err(S3Error::GetFailed(svc_err.to_string()))
                }
            }
        }
    }

    /// Check whether a book exists in S3 without downloading it.
    ///
    /// Returns `true` if the object exists, `false` if it does not.
    ///
    /// # Errors
    ///
    /// Returns [`S3Error::HeadFailed`] for any error other than `NotFound`.
    #[instrument(skip(self), fields(bucket = %self.bucket))]
    pub async fn exists(&self, cid_bytes: &[u8; 32]) -> Result<bool, S3Error> {
        let key = self.object_key(cid_bytes);
        debug!(key, "checking book existence");

        let result = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await;

        match result {
            Ok(_) => Ok(true),
            Err(sdk_err) => {
                let svc_err = sdk_err.into_service_error();
                if svc_err.is_not_found() {
                    Ok(false)
                } else {
                    Err(S3Error::HeadFailed(svc_err.to_string()))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_library(prefix: &str) -> S3Library {
        // Construct directly without AWS — safe for unit tests that only
        // exercise pure logic like key formatting.
        let config = aws_sdk_s3::config::Builder::new()
            .region(aws_config::Region::new("us-east-1"))
            .behavior_version(BehaviorVersion::latest())
            .build();
        S3Library {
            client: Client::from_conf(config),
            bucket: "test-bucket".into(),
            prefix: prefix.into(),
        }
    }

    #[test]
    fn object_key_format() {
        let lib = make_library("harmony/");
        let cid = [0xabu8; 32];
        let key = lib.object_key(&cid);
        assert_eq!(
            key,
            format!("harmony/book/{}", "ab".repeat(32)),
            "key must follow {{prefix}}book/{{hex}} format"
        );
    }

    #[test]
    fn object_key_different_prefix() {
        let lib = make_library("prod/v2/");
        let cid = [0x00u8; 32];
        let key = lib.object_key(&cid);
        assert!(
            key.starts_with("prod/v2/book/"),
            "key must include the configured prefix"
        );
        assert_eq!(
            key,
            format!("prod/v2/book/{}", "00".repeat(32)),
        );
    }
}
