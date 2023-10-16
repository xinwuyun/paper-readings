n = open("目录.md", "w")
with open("SUMMARY.md") as f:
    text = f.readlines()
    for line in text:
        i = line.split("](")
        if len(i) != 2:
            n.write(line+"\n")
            continue
        title = i[0].split("[")[1]
        link = i[1][:-2]
        
        j = line.split("[")
        new = "[[%s|%s]]" % (link, title)
        n.write(j[0]+new+"\n")
n.close()
