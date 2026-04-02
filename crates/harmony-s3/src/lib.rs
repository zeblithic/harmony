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

#[cfg(feature = "archivist")]
pub mod archivist;

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
    /// When true, use a custom endpoint (e.g., Cloudflare R2) that may not
    /// support all S3 storage classes. Omits INTELLIGENT_TIERING on PUTs.
    custom_endpoint: bool,
}

impl S3Library {
    /// Create a new [`S3Library`] client.
    ///
    /// Configuration is loaded from the environment (AWS_REGION,
    /// AWS_ACCESS_KEY_ID, etc.) with `region` overriding the ambient region.
    ///
    /// Set `endpoint` to point at an S3-compatible provider (e.g., Cloudflare
    /// R2: `https://<account-id>.r2.cloudflarestorage.com`). When `None`, the
    /// SDK uses the default AWS S3 endpoints.
    ///
    /// # Errors
    ///
    /// Returns [`S3Error::ConfigError`] if the AWS SDK cannot be configured.
    #[instrument(skip_all)]
    pub async fn new(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        region: Option<String>,
        endpoint: Option<String>,
    ) -> Result<Self, S3Error> {
        let bucket = bucket.into();
        let mut prefix = prefix.into();
        // Normalize prefix: ensure trailing slash (or empty).
        if !prefix.is_empty() && !prefix.ends_with('/') {
            prefix.push('/');
        }

        let mut config_loader = aws_config::defaults(BehaviorVersion::latest());
        // Only set region explicitly if configured. Otherwise, let the SDK
        // auto-detect from AWS_DEFAULT_REGION, ~/.aws/config, etc.
        if let Some(ref region) = region {
            config_loader = config_loader.region(aws_config::Region::new(region.clone()));
        }
        let custom_endpoint = endpoint.is_some();
        if let Some(ref endpoint) = endpoint {
            config_loader = config_loader.endpoint_url(endpoint);
        }
        let config = config_loader.load().await;
        let client = Client::new(&config);

        debug!("S3Library initialised");
        Ok(Self {
            client,
            bucket,
            prefix,
            custom_endpoint,
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
    /// On AWS S3, uses `INTELLIGENT_TIERING` storage class so infrequently
    /// accessed books migrate to cheaper tiers automatically. On custom
    /// endpoints (e.g., Cloudflare R2), the storage class is omitted since
    /// R2 only supports STANDARD.
    ///
    /// The call is idempotent — re-uploading the same content identifier
    /// overwrites with identical data.
    ///
    /// # Errors
    ///
    /// Returns [`S3Error::PutFailed`] if the upload fails.
    #[instrument(skip(self, data), fields(bucket = %self.bucket))]
    pub async fn put_book(&self, cid_bytes: &[u8; 32], data: Vec<u8>) -> Result<(), S3Error> {
        let key = self.object_key(cid_bytes);
        debug!(key, bytes = data.len(), "putting book");

        let mut req = self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(ByteStream::from(data));
        if !self.custom_endpoint {
            req = req.storage_class(StorageClass::IntelligentTiering);
        }
        req.send()
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
            custom_endpoint: false,
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
