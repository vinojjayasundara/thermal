import os
import time
import subprocess


class Ops(object):
    def __init__(self):
        self.filename = 'test.txt'
        # Creating a dummy file
        with open(self.filename, 'w') as fp:
            fp.write('00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
                     '00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')

    def write_file(self, duration=60):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        timeout = time.time() + duration
        with open(self.filename, 'w') as fp:
            while True:
                if time.time() > timeout:
                    break
                fp.write('00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
                         '00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')

    def read_file(self, duration=60):
        timeout = time.time() + duration
        while True:
            if time.time() > timeout:
                break
            with open(self.filename, 'r') as fp:
                _ = fp.readlines()

    def cpu_stress(self, cpu=.5, duration=60):
        subprocess.call(f'python3 -m cpu_load_generator -l {cpu} -d {duration} -c 0'.split(' '))

    def memory_stress(self, mem=128, duration=60):
        subprocess.call(f'stress -c 1 -i 1 -m 1 --vm-bytes {mem}M -t {duration}s'.split(' '))


class Normal(object):
    def __init__(self):
        self.ops = Ops()

    def seq1(self):
        self.ops.read_file(10)
        self.ops.write_file(10)
        self.ops.cpu_stress(.5, 20)
        self.ops.write_file(20)

    def seq2(self):
        self.ops.write_file(20)
        self.ops.memory_stress(duration=15)
        self.ops.cpu_stress(duration=15)
        self.ops.read_file(10)

    def seq3(self):
        self.ops.cpu_stress(.5, 30)
        self.ops.memory_stress(duration=30)

    def seq4(self):
        self.ops.cpu_stress(.5, 30)
        self.ops.write_file(30)

    def seq5(self):
        self.ops.read_file(30)
        self.ops.memory_stress(duration=30)


class Abnormal(object):
    def __init__(self):
        self.ops = Ops()

    def seq1(self):
        self.ops.read_file(10)
        self.ops.write_file(10)
        self.ops.cpu_stress(.5, 20)
        self.ops.read_file(10)
        self.ops.cpu_stress(.95, 20)
        self.ops.write_file(20)

    def seq2(self):
        self.ops.write_file(20)
        self.ops.memory_stress(duration=15)
        self.ops.cpu_stress(duration=15)
        self.ops.memory_stress(duration=10)
        self.ops.cpu_stress(cpu=.95, duration=20)
        self.ops.read_file(10)

    def seq3(self):
        self.ops.cpu_stress(.5, 30)
        self.ops.write_file(15)
        self.ops.memory_stress(duration=30)

    def seq4(self):
        self.ops.cpu_stress(.5, 30)
        self.ops.write_file(30)
        self.ops.memory_stress(15)

    def seq5(self):
        self.ops.read_file(30)
        self.ops.cpu_stress(cpu=.95, duration=20)
        self.ops.memory_stress(duration=30)


seq = Normal()
seq.seq1()
