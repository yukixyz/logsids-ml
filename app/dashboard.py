import io, base64, matplotlib.pyplot as plt, pandas as pd
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    return enc
def top_ips_figure(df, top_n=10):
    agg = df.groupby('source_ip').size().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10,4))
    agg.plot.bar(ax=ax)
    ax.set_xlabel('Source IP')
    ax.set_ylabel('Events')
    ax.set_title('Top IPs by events')
    fig.tight_layout()
    return fig
def timeseries_figure(df, ip=None):
    df = df.copy()
    df['minute'] = df['timestamp'].dt.floor('T')
    if ip:
        df = df[df['source_ip'] == ip]
    agg = df.groupby('minute')['anomaly_score'].mean()
    fig, ax = plt.subplots(figsize=(12,4))
    agg.plot(ax=ax)
    ax.set_title('Average anomaly score per minute' + (f' - {ip}' if ip else ''))
    ax.set_ylabel('Score')
    fig.tight_layout()
    return fig
def make_dashboard_html(df, ip=None):
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    fig1 = top_ips_figure(df)
    fig2 = timeseries_figure(df, ip=ip)
    b1 = fig_to_base64(fig1)
    b2 = fig_to_base64(fig2)
    html = f"""<html><head><title>log-ids-ml dashboard</title></head><body><h1>log-ids-ml Dashboard</h1><h2>Top IPs</h2><img src='data:image/png;base64,{b1}'/><h2>Timeseries</h2><img src='data:image/png;base64,{b2}'/></body></html>"""
    return html
