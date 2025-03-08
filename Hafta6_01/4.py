import re
xxx = "Ahmet al renkli bir şal aldı."

# tüm al ifadelerinin listesi
print(re.findall("al", xxx))

# şal ifadesini ara
print(re.search("şal", xxx))

# “al” a göre böl
print(re.split("al", xxx)) 

# Boşlukları zzz yap
print(re.sub(" ", "zzz", xxx)) 

