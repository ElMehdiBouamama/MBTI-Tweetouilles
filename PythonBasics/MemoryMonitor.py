import time
import psutil
import sys

toolbar_width = 76
sys.stdout.write("    MEMORY USAGE \n")
sys.stdout.write("[%s]\n" % ("-" * toolbar_width))
sys.stdout.write("    CPU USAGE \n")
sys.stdout.write("[%s]" % ("-" * toolbar_width))
while(1):
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    cpu_hash = int(cpu_percent * toolbar_width / 100)
    memory_hash = int(memory_percent * toolbar_width / 100)
    sys.stdout.write("\r" + "[%s%s]%s" % ("#" * cpu_hash, ("-" * (toolbar_width - cpu_hash)), cpu_percent))
    sys.stdout.write("\r\b\r\b")
    sys.stdout.write("\r" + "[%s%s]%s" % ("#" * memory_hash, ('-' * (toolbar_width - memory_hash)), memory_percent))
    sys.stdout.write("\n\n")
    time.sleep(0.5)
