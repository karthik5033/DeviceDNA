from scapy.all import sniff, TCP, IP
def handle(pkt):
    if TCP in pkt and pkt[TCP].dport == 8000:
        print(pkt.summary())
        return True
sniff(prn=handle, stop_filter=handle)
