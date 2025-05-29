from langdetect import detect
import json

num = 0
with open("hotel_reviews_train_en.JSON", "w") as out_file:
    with open("hotel_reviews_train.JSON","r") as in_file:
        lines = in_file.readlines()
        for line in lines:
            num += 1
            if num%100==0: print(num)
            json_line = json.loads(line)
            try:
                lang = detect(json_line["text"])
                if lang == "en":
                    out_file.write(line)
            except:
                print("exception")
                pass


