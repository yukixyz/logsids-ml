#!/usr/bin/env python3
import csv, random, argparse
from datetime import datetime, timedelta

def random_private_ip():
    return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def random_public_ip():
    parts = [random.randint(1,223) for _ in range(4)]
    return ".".join(map(str, parts))

METHODS = ['GET','POST','HEAD','PUT','DELETE']
PATHS = ['/', '/login', '/admin', '/api/data', '/wp-admin', '/search?q=test', '/config.php', '/.env', '/api/auth', '/api/v1/items']
UAS = ['Mozilla/5.0', 'curl/7.68.0', 'sqlmap/1.4', 'Mozilla/5.0 (bot)', 'python-requests/2.31.0']

def gen_line(ts, ip, method, path, status, ua):
    return [ts.strftime("%Y-%m-%d %H:%M:%S"), ip, method, path, str(status), ua]

def main(output='sample_logs.csv', lines=5000, attack_rate=0.005):
    start = datetime.utcnow()
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp','source_ip','method','path','status','user_agent'])
        for i in range(lines):
            ts = start + timedelta(seconds=i//5)
            if random.random() < attack_rate:
                ip = "203.0.113.55"
                path = random.choice(PATHS + ['/etc/passwd','/admin/login','/wp-login.php','/phpmyadmin/'])
                method = random.choice(['GET','POST'])
                status = random.choice([200,401,403,404,500])
                ua = 'sqlmap/1.4' if random.random() < 0.6 else 'Mozilla/5.0 (bot)'
            else:
                ip = random_private_ip() if random.random() < 0.7 else random_public_ip()
                path = random.choice(PATHS)
                method = random.choice(METHODS)
                status = random.choices([200,404,401,500], weights=[0.8,0.1,0.06,0.04])[0]
                ua = random.choice(UAS)
            writer.writerow(gen_line(ts, ip, method, path, status, ua))
    print(f"Generated {lines} lines to {output}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='sample_logs.csv')
    p.add_argument('--lines', type=int, default=5000)
    p.add_argument('--attack-rate', type=float, default=0.005)
    args = p.parse_args()
    main(output=args.output, lines=args.lines, attack_rate=args.attack_rate)
