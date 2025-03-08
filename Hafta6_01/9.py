import re
txt1 = "Ahmet al renkli bir şal aldı."
txt2 = "Mehmet kırmızı 12312 top 2789456 getirdi."

bulunan1 = re.search("\d{10}", txt2)
print(bulunan1)