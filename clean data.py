def munger (data):
    for index, row in data.iterrows ()
        text = row['text']
        text = re.sub("@","",text)
        text = re.sub("#","",text)
        text = re.sub("bit\.ly.*\s?","",text)
        text = re.sub("instagr\.am.*\s?","",text)
        text = re.sub("https?:.*\s?","",text)
        text = re.sub("t\.co.*\s?","",text)
        text = re.sub("pic\.twitter\.com\S*\s?","",text)
        data.set_value(index,"text",text)
