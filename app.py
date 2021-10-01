
from flask import Flask, request, url_for, redirect, render_template
from flask_mail import Mail,Message
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,MBartForConditionalGeneration, MBart50TokenizerFast
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_metric



app = Flask(__name__)

PATH = 'D:/venus/knight/model/TransBart/'
model = MBartForConditionalGeneration.from_pretrained(PATH, local_files_only=True)
tokenizer = MBart50TokenizerFast.from_pretrained(PATH, local_files_only=True, src_lang="en_XX")

sacrebleu = load_metric('sacrebleu')

app.config['MAIL_SERVER'] ='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = "rakshitkumarkn@gmail.com"
app.config['MAIL_PASSWORD'] = "rakgo2260"
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail =  Mail(app)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/trans", methods=['POST', 'GET'])
def trans():
    if request.method == 'POST':
        try:      
            text_to_translate = request.form["text-to-translate"].lower()
            selected_language = request.form["select-language"]
            tokenized_text = tokenizer(text_to_translate, return_tensors='pt')
            translation = model.generate(**tokenized_text, forced_bos_token_id=tokenizer.lang_code_to_id[selected_language])
            translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
            text = translated_text
            sacrebleu = load_metric('sacrebleu')
            predictions = [text]
            text_to_translate2 = request.form["text-to-translate2"].lower()
            references = [[text_to_translate2]]  
            results = sacrebleu.compute(predictions=predictions, references=references)
            text2 = results["score"]
            text3 = results["counts"]
            text4 = results["precisions"]
        except:
                text = "{ERROR: We are not able to handle your request right now}"
                text2 = "{error}"
        return render_template('trans.html', translation_result=text, translation_result2=text2, translation_result3=text3, translation_result4=text4)
    return render_template("trans.html")

@app.route("/Seq", methods=['POST', 'GET'])
def Seq():
    if request.method == 'POST':
        try:      
            text_to_translate = request.form["text-to-translate"].lower()
            selected_language = request.form["select-language"]
            tokenized_text = tokenizer(text_to_translate, return_tensors='pt')
            translation = model.generate(**tokenized_text, forced_bos_token_id=tokenizer.lang_code_to_id[selected_language])
            translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
            text = translated_text
            sacrebleu = load_metric('sacrebleu')
            predictions = [text]
            text_to_translate2 = request.form["text-to-translate2"].lower()
            references = [[text_to_translate2]]  
            results = sacrebleu.compute(predictions=predictions, references=references)
            text2 = results["score"]
            text3 = results["counts"]
            text4 = results["precisions"]
        except:
                text = "{ERROR: We are not able to handle your request right now}"
                text2 = "{error}"
        return render_template('Seq.html', translation_result=text, translation_result2=text2,  translation_result3=text3, translation_result4=text4)
    return render_template("Seq.html")

@app.route("/bert", methods=['POST', 'GET'])
def bert():
    if request.method == 'POST':
        try:      
            text_to_translate = request.form["text-to-translate"].lower()
            selected_language = request.form["select-language"]
            tokenized_text = tokenizer(text_to_translate, return_tensors='pt')
            translation = model.generate(**tokenized_text, forced_bos_token_id=tokenizer.lang_code_to_id[selected_language])
            translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
            text = translated_text
            sacrebleu = load_metric('sacrebleu')
            predictions = [text]
            text_to_translate2 = request.form["text-to-translate2"].lower()
            references = [[text_to_translate2]]  
            results = sacrebleu.compute(predictions=predictions, references=references)
            text2 = results["score"]
            text3 = results["counts"]
            text4 = results["precisions"]
        except:
                text = "{ERROR: We are not able to handle your request right now}"
                text2 = "{error}"
        return render_template('bert.html', translation_result=text, translation_result2=text2, translation_result3=text3, translation_result4=text4)
    return render_template("bert.html")   
   
@app.route("/smt", methods=['POST', 'GET'])
def smt():
    if request.method == 'POST':
        try:      
            text_to_translate = request.form["text-to-translate"].lower()
            selected_language = request.form["select-language"]
            tokenized_text = tokenizer(text_to_translate, return_tensors='pt')
            translation = model.generate(**tokenized_text, forced_bos_token_id=tokenizer.lang_code_to_id[selected_language])
            translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
            text = translated_text
            sacrebleu = load_metric('sacrebleu')
            predictions = [text]
            text_to_translate2 = request.form["text-to-translate2"].lower()
            references = [[text_to_translate2]]  
            results = sacrebleu.compute(predictions=predictions, references=references)
            text2 = results["score"]
            text3 = results["counts"]
            text4 = results["precisions"]
        except:
                text = "{ERROR: We are not able to handle your request right now}"
                text2 = "{error}"
        return render_template('smt.html', translation_result=text, translation_result2=text2, translation_result3=text3, translation_result4=text4)
    return render_template("smt.html")

@app.route('/home')
def home():
    return render_template('cont.html')

@app.route('/send_message', methods=['GET','POST'])
def send_message():
    if request.method == 'POST':
        email = request.form['email']
        subject = request.form['subject']
        msg = request.form['message']

        message = Message(subject, sender="rakshitkumarkn@gmail.com",recipients=[email])

        message.body = msg

        mail.send(message)

        success = "Your Message has been sent"

        return render_template('cont.html',success=success)

@app.route("/team")
def team():
    return render_template('team.html')

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)