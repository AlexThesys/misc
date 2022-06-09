import r2pipe, sys
import pdb

magic = "ff88ff88ff88ff88"
num_cycles = 16
num_funcs = 2
seeds = [ 0x7facacac, 0x00bdbdbd ]

class simple:
    holdrand = 0
    def seed(self, s):
        self.holdrand = s
    def i(self, max):
        self.holdrand = (self.holdrand * 214013 + 2531011)  & 0xFFFFFFFFFFFFFFFF
        return ((self.holdrand >> 16) & 0x7fff) % max

def cypher_func(r, k):
    return ((r * r) % (k + r)) & 0xFF
    
def encrypt(buf, keys):
    size = len(buf)
    temp = [0] * size
    for i in range(num_cycles):
        for j in range(0, size, 2):
            temp[j//2] = buf[j]
            buf[j] = buf[j+1]
            buf[j+1] = temp[j//2] ^ cypher_func(buf[j+1], keys[i])
 
if len(sys.argv) < 2:
    raise Exception("Please provide the path to the file!")
 
keys_foo = []
keys_bar = []
rnd_gen = simple()
rnd_gen.seed(seeds[0])
for _ in range(num_cycles):
    keys_foo.append(rnd_gen.i(0xff))
rnd_gen.seed(seeds[1])
for _ in range(num_cycles):
    keys_bar.append(rnd_gen.i(0xff))
 
keys = [keys_foo, keys_bar]
r2 = r2pipe.open(sys.argv[1], ["-w"])
r2.cmd("/x " + magic)
line = str(r2.cmd("s hit0_0"))
addr = int(line.split()[0], 16) - 0x2
r2.cmd("af @ " + hex(addr))
json = r2.cmdj("pdj " + str(num_funcs) + " @ " + hex(addr+0xa+0x3+0x5+0x3+0x3))
func_addr = []
for d in json:
    for key, value in d.items():
        if key == "disasm":
            beg = value.find("[")
            end = value.find("]")
            substr = value[beg+1:end]
            print("Function address = " + substr)
            func_addr.append(int(substr, 16))
if (len(func_addr) != num_funcs):
    print("Not all the functions found!")
    exit()
r2.cmd("wx 90909090909090909090 @ " + hex(addr))
r2.cmd("wx 9090909090 @ " + hex(addr + 0xA + 0x3))
r2.cmd("wx 909090 @ " + hex(addr+0xa+0x3+0x5+0x3))
k = 0
for fa in func_addr:
    r2.cmd("af @ " + hex(fa))
    json = r2.cmdj("afij @ " + hex(fa))
    for d in json:
        for key, value in d.items():
            if key == "size":
                print("Function size = " + hex(value))
                data = r2.cmdj("pxj " + str(value) + " @ " + hex(fa))
                encrypt(data, keys[k])
                k += 1
                hex_data = bytearray(b'')
                for d in data:
                    hex_data.append(d)
                hex_str = hex_data.hex()
                print("Encrypted hex string written: " + hex_str)
                r2.cmd("wx " + hex_str + " @ " + hex(fa))
r2.quit()
    
    
