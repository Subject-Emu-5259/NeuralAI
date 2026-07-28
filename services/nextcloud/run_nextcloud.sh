#!/usr/bin/env bash
# NeuralDrive (Nextcloud) runtime launcher
set -u
NC=/home/workspace/Projects/NeuralAI/services/nextcloud

# Clean stale sockets/pids
rm -f /run/php/php8.2-fpm.sock 2>/dev/null
mkdir -p /run/php /var/run/apache2 /var/lock/apache2
chown -R www-data:www-data "$NC/data" 2>/dev/null || true

# Start php-fpm (TCP 127.0.0.1:9000 per pool config)
php-fpm8.2 -D -F 2>/dev/null || php-fpm8.2 &
sleep 2

# Start Apache in foreground (Zo service supervisor keeps it alive)
exec apache2ctl -D FOREGROUND
