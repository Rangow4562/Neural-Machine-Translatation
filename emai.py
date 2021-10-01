PATH = 'D:/venus/knight/model/TransBart/'
model = MBartForConditionalGeneration.from_pretrained(PATH, local_files_only=True)
tokenizer = MBart50TokenizerFast.from_pretrained(PATH, local_files_only=True, src_lang="en_XX")
PATH2 = 'D:/venus/knight/model/Seq2SeqBart/'
model2 = MBartForConditionalGeneration.from_pretrained(PATH2, local_files_only=True)
tokenizer2 = MBart50TokenizerFast.from_pretrained(PATH2, local_files_only=True, src_lang="en_XX")
PATH3 = 'D:/venus/knight/model/BERTBART/'
model3 = MBartForConditionalGeneration.from_pretrained(PATH3, local_files_only=True)
tokenizer3 = MBart50TokenizerFast.from_pretrained(PATH3, local_files_only=True, src_lang="en_XX")
PATH4 = 'D:/venus/knight/model/SmtBART/'
model4 = MBartForConditionalGeneration.from_pretrained(PATH4, local_files_only=True)
tokenizer4 = MBart50TokenizerFast.from_pretrained(PATH4, local_files_only=True, src_lang="en_XX")