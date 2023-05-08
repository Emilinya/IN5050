import sys
import math

try:
    with open(sys.argv[1], "r") as infile:
        times = [float(line) for line in infile]
    avg = sum(times) / len(times)
    std = math.sqrt(sum((time - avg)**2 for time in times)) / len(times)
    with open(sys.argv[1].replace("temp", "runtimes"), "w") as outfile:
        outfile.write(f"""\
Runtime data from 10 runs. Each run encoded 100 frames of a 352x288 video.
avg ± std [s] | min [s] | max [s]
{avg:.6f} ± {std:.6f} | {min(times):.6f} | {max(times):.6f}

Data:
{' '.join(f'{time:.6f}' for time in times)}
""")
except Exception as e:
    print(e)
