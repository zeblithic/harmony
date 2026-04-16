use std::env;
use std::sync::Arc;
use std::time::Duration;

use harmony_mail_discovery::cache::SystemTimeSource;
use harmony_mail_discovery::dns::HickoryDnsClient;
use harmony_mail_discovery::http::ReqwestHttpClient;
use harmony_mail_discovery::resolver::{DefaultEmailResolver, EmailResolver, ResolverConfig};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    tracing_subscriber::fmt::init();
    let arg = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: harmony-mail-discovery-debug <user@domain>");
        std::process::exit(2);
    });
    let (local, domain) = match arg.split_once('@') {
        Some(p) => p,
        None => {
            eprintln!("expected user@domain, got: {arg}");
            std::process::exit(2);
        }
    };

    let dns: Arc<dyn harmony_mail_discovery::dns::DnsClient> =
        Arc::new(HickoryDnsClient::from_system_or_google(Duration::from_secs(5)));
    let http: Arc<dyn harmony_mail_discovery::http::HttpClient> = Arc::new(
        ReqwestHttpClient::new(Duration::from_secs(5), Duration::from_secs(10), 1_000_000)
            .expect("build reqwest client"),
    );
    let time: Arc<dyn harmony_mail_discovery::cache::TimeSource> = Arc::new(SystemTimeSource);

    let resolver = DefaultEmailResolver::new(dns, http, time, ResolverConfig::default());
    let outcome = resolver.resolve(local, domain).await;
    println!("{outcome:?}");
}
