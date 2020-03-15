import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    print(some_string)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
print('Press Ctrl+C')

if __name__ == '__main__':
    some_string = "aldo"
    while True:
        pass