# Neural-Machine-Translatation
* Businesses all around the world rely on translation services as an essential component of their daily operations. Translation helps organizations function easily and efficiently across international borders results in increased sales.
* An increase in humans' dependence on computer-aided systems has pushed me to work on more effective communication technologies that simulate interactions as well as natural languages translation. Over the last few years, translation has become a hotly debated academic topic. So, I'll primarily concentrate on four use case algorithms in this research namely: Transformer, Seq2Seq with Attention, BERT, and Statistical Machine Translation techniques.
* The translation is required for the transmission of information, knowledge, and ideas. Effective and empathetic communication across various cultures is critical. Let us begin with a single model. In machine translation software, it would take a sentence in one language and translate it into another. Because the Seq2Seq Attention Model uses both Recurrent Attention and Transformer, it may quickly learn the most likely decoding sequence in the target language. The capabilities for fine-tuning data to a given linguistic environment and resolving problems encountered on a daily basis. This deep learning model is extremely intriguing in terms of delivering a high-performing model.
* Machine translations are not good for lengthy periods. The accuracy percentage of automated translations ranges from 70% to 90%, but seldom exceeds 100%. It's unusual to come across a digital gadget that can properly translate each word. Human labor is recommended for the highest accuracy.  Although human translation appears to outperform artificial intelligence, it is far from perfect. The primary issue is the delayed speed. When compared to translation technologies, humans take far longer to produce translations. Human translation appears to be considerably more sophisticated than AI as the best option for translating context, although AI is improving every day. To achieve the ideal balance of accuracy and quality in translation, any good consumer of translation will consider both machine and human translating technologies.

## Dataset
### WMT: Dataset (WMT 2012)
Link: https://nlp.stanford.edu/projects/nmt/
The above website includes information on the most recent neural machine translation (NMT) dataset at the Stanford NLP group. They release a codebase, which provides cutting-edge outcomes in a variety of translation jobs, including English-German and English-Spanish, etc. In addition, we share the preprocessed data that we used to train NLP models, as well as a trained dataset that is easily useable with the codebase, to encourage reproducibility and transparency. This dataset is usually used in Statistical Machine Translation.

### OPUS: Dataset (OPUS 100)
Link: https://opus.nlpl.eu/
OPUS is a growing collection of web-translated literature. The OPUS project aims to convert and align freely available internet data, add linguistic annotation, and offer the community with a publicly available parallel corpus. OPUS is built on open source technologies, and the corpus is likewise available as an open content package. They assembled the present collection using a variety of tools. All of the pre-processing is handled automatically. There have been no manual corrections.

### OPUS: Dataset (TED2019)
Link: https://opus.nlpl.eu/TED2019.php
This dataset contains a crawl of nearly 4000 TED and TED-X transcripts from July 2019. The transcripts have been translated by a global community of volunteers (https://www.ted.com/participate/translate) to more than 100 languages.
The transcripts for the different languages have been sentence aligned to generate a parallel corpus that can be used to train machine translation systems.

### OPUS: Dataset (wiki v20210402)
Link: https://opus.nlpl.eu/wikimedia.php
Wikipedia translations published by the wiki foundation and their article translation system. The parallel data sets are published at https://dumps.wikimedia.org/other/contenttranslation
NEW: additional sentence alignment to avoid long segments in translation units
306 languages, 2,575 bitexts
total number of files: 306
total number of tokens: 918.05M
total number of sentence fragments: 31.62.

### Models
I have chosen four distinct models in this project, notably: -
*	Transformer (NLP)
*	Sequence to Sequence with Attention
*	Bidirectional Encoder Representations from Transformers
*	Statistical Machine Translation
