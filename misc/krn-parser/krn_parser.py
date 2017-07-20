import numpy as np
import sys
import os

pc_dict = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
p_alt = {"#": 1, "n": 0, "-": -1}
extras = ["{", "}", "x"]
unknown = []
lengths = []

def parse_files(source, dest):
    files = os.listdir(source)
    
    for f in files:
        path = source + f
        parsed = parse_file(path)
        write_file(f, parsed, dest)

    print("Done.")
    

def parse_file(path):
    parsed_file = []
    with open(path, "r", 0) as f:
        in_mel = False
        mel_over = False
        line_val = None
        mel = []
        for line in f:
            line = line [:-1]
            if mel_over:
                line_val = parse_info(line)
            elif in_mel == False and mel_over == False and (line[0] == "!" or line[0] == "*"):
                line_val = parse_info(line)
            elif in_mel == False and mel_over == False and line[0] != "!" and line[0] != "*":
                in_mel == True
                line_val = parse_note(line)
                mel.append(line_val)
                continue
            elif in_mel and ("==" in line):
                mel_over = True
                in_mel = False
               
            if line_val != None:
                parsed_file.append(line_val)
                
        mel = build_mel(mel)
        parsed_file += mel
            
    return parsed_file
                
    
def write_file(f_name, parsed, dest):
    path = dest + "/" + f_name
    f = open(path, 'w+')
    for line in parsed:
        f.write(line)
        f.write("\n")
    f.close()
    
    
def parse_info(line):
    val = None
    if "*M" in line:
        val = "*T " + line[2:]
    elif "*" in line and ":" in line:
        key = line[1:-1]
        key_alt = ""
        if len(key) == 2:
            key_alt = key[1]
            if key_alt == "-":
                key_alt = "-flat"
            else:
                key_alt = "-sharp"
            key = key[0]
        mode = "Minor"
        if key[0].isupper():
            mode = "Major"
        key = key.upper()
        val = "*K " + key + key_alt + " " + mode
    elif "!!!AMT" in line:
        s = line.split()
        val = "*M " + str(s[1]) + " " 
        if len(s) ==3:
            val += str(s[2])
    return val


def parse_note(line):
    return_val = None
    pc = None # pitch class
    pitch_alt = 0 # semi-tone alteration
    octave = 4.5 
    length = "" # kern rhythm value
    length_alt = 0 # number of dots on rhythm
    barline = None 
    rest = False
    tie = 0
    
    i = 0
    while True:
        if line[i] == "=": # barline
            barline = [line[i], line[i+1]]
            break
        elif is_int(line[i]): # rhythm value
            length += str(line[i])
        elif line[i] == ".": # dotted rhythm
            if (len(line) > (i + 1)) and line[i+1] == "5":
                length += ".5"
                i += 1
            else:
                length_alt += 1
        elif (line[i].upper()) in pc_dict: #pitch
            pc = pc_dict[line[i].upper()]
            if line[i].isupper():
                octave = np.round(octave - 1)
            else:
                octave = int(octave + 1)
        elif line[i] in p_alt: # sharps/flats
            pitch_alt += p_alt[line[i]]
        elif line[i] == "r": # rest
            rest = True
        elif line[i] == "[": # start tie
            tie = 1 
        elif line[i] == "]": # end tie
            tie = -1
        elif line[i] in extras: # unecessary character
            1 + 1
        else: # unknown character
            if (line[i] in unknown) == False:
                unknown.append(line[i])
            print line[i]
            
        i += 1
        if i >= len(line):
            break
            
    if barline != None:
        return barline
    
    pitch = None
    rhythm = None
    
    # calculate pitch
    if rest:
        pitch = "r"
    else:
        pitch = (octave * 12) + pc + pitch_alt
    
    # calculate rhythm
    if int(float(length)) == 0:
        length = 0.5
    length = 4000.0 / float(length)
    rhythm = length
    for dot in range(length_alt):
        rhythm += np.power(0.5, (dot + 1)) * (length)
        
    length = int(length)
    
    # debug
    if (length in lengths) == False:
        lengths.append(length)
        
    return [pitch, rhythm, tie]                
                
    
def build_mel(mel):
    build = []
    
    # calculate pickup length
    pickup = 0.0
    for element in mel:
        if element[0] == "=":
            break
        else:
            pickup += element[1]
    build.append("*P " + str(round(pickup)))
    
    time = 0
    tie_length = 0
    
    for i in range(len(mel)):
        note = ""
        if mel[i][0] == "=": # barline
            continue
        elif mel[i] [0] == "r": # rest
            time += int(mel[i][1])
            continue
        elif mel[i][2] == 1: # tie begins
            tie_length += mel[i][1]
            continue
        else: # note
            pitch = mel[i][0]
            start = time
            time += mel[i][1] + tie_length
            end = time
            note += "Note\t" + str(round(start)) + "\t" + str(round(end)) + "\t" + str(pitch)
            
        if mel[i][2] == -1: # tie ends:
            tie_length = 0
            
        build.append(note)
    
    return build


def is_int(i):
    try: 
        int(i)
        return True
    except ValueError:
        return False
             
        
def main():
    parse_files(sys.argv[1], sys.argv[2])
    print("Unknown: ", unknown)
    print("Lengths: ", lengths)
    
main()
