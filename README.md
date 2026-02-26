# 📚 نظام Q&A للمستندات
> نظام ذكي للأسئلة والأجوبة مبني على RAG (Retrieval-Augmented Generation) مع دعم البث الفوري واكتشاف اللغة

---

## 🏗️ معمارية النظام

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Layer                       │
│              Gradio UI — app/main.py (Port 7860)        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 LLM Pipeline Layer                      │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Ingestion  │  │  Retrieval   │  │    Answer Gen  │  │
│  │  Pipeline   │  │  Pipeline    │  │    Pipeline    │  │
│  │             │  │              │  │                │  │
│  │ Load → Split│  │ Embed Query  │  │ Build Prompt   │  │
│  │ → Embed     │  │ → MMR Search │  │ → Stream LLM   │  │
│  │ → Store     │  │ → Top-K      │  │ → Citations    │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   Storage Layer                         │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │File Storage │  │  Vector DB   │  │  Chat Memory   │  │
│  │data/uploads │  │    FAISS     │  │  RAM + Session │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 التثبيت والتشغيل

### المتطلبات
- Python 3.10+
- 4GB RAM على الأقل
- مفتاح [OpenRouter API](https://openrouter.ai) (مجاني متاح)

### الخطوات

**الطريقة السريعة (Windows):**
```bat
setup.bat
```
السكريبت هيعمل كل حاجة تلقائياً: venv، تثبيت المكتبات، المجلدات، وإنشاء `.env`

**أو يدوياً:**
```bash
# 1. تثبيت المكتبات
pip install -r requirements.txt

# 2. إعداد المتغيرات البيئية
copy .env.example .env
# عدّل .env وأضف OPENROUTER_API_KEY

# 3. تشغيل الواجهة
python app\main.py
```

افتح المتصفح على: **http://localhost:7860**

---

## 📁 هيكل المشروع

```
project/
├── app/
│   ├── main.py          # الواجهة الرئيسية (Gradio) + كل منطق RAG
│   ├── api.py           # REST API (FastAPI) — اختياري
│   ├── evaluator.py     # تقييم جودة الإجابات
├── data/
│   ├── uploads/         # الملفات المرفوعة
│   ├── vectordb/        # قاعدة بيانات FAISS (حسب session)
│   └── cache/           # تخزين مؤقت للـ embeddings
├── .env                 # المتغيرات البيئية (لا ترفعه على GitHub)
├── .env.example         # قالب المتغيرات البيئية (آمن للرفع)
├── requirements.txt
├── setup.bat            # سكريبت الإعداد (Windows)
└── README.md
```

---

## ⚙️ متغيرات البيئة

| المتغير | القيمة الافتراضية | الوصف |
|---------|------------------|--------|
| `OPENROUTER_API_KEY` | — | مفتاح OpenRouter API **(مطلوب)** |
| `base_url` | `https://openrouter.ai/api/v1` | رابط الـ API |
| `OPENROUTER_MODEL` | `stepfun/step-3.5-flash:free` | اسم النموذج |
| `CHUNK_SIZE` | `1000` | حجم القطعة النصية |
| `CHUNK_OVERLAP` | `200` | التداخل بين القطع |
| `TOP_K_RESULTS` | `4` | عدد النتائج المسترجعة |
| `MAX_FILE_SIZE_MB` | `50` | أقصى حجم للملف |
| `PORT` | `7860` | منفذ Gradio |

---

## 🌟 المميزات

| الميزة | التفاصيل |
|--------|----------|
| 🌍 **اكتشاف اللغة** | يرد بالعربية أو الإنجليزية تلقائياً حسب سؤالك |
| ⚡ **بث فوري** | الإجابات تظهر token by token |
| 🧮 **LaTeX** | دعم كامل لعرض المعادلات الرياضية |
| 💬 **ذاكرة المحادثة** | يتذكر آخر 10 رسائل لكل session |
| 📄 **أنواع الملفات** | PDF · DOCX · TXT |
| 🔍 **MMR Search** | بحث متنوع يتجنب التكرار |
| 📊 **تقييم تلقائي** | يقيّم جودة الإجابات ويحفظ التقرير |
| 🔧 **استعادة الجلسة** | يعيد تحميل الـ vectorstore من الديسك عند الحاجة |

---

## 📊 مؤشرات الأداء

| المؤشر | الهدف | الحالة |
|--------|-------|--------|
| رفع الملف ومعالجته | ≤ 3s | ✅ |
| الرد الكامل (بث) | ≤ 5s | ✅ |
| Retrieval (MMR) | < 1s | ✅ |
| دقة الإجابة | > 80% | يعتمد على النموذج |

---

## 🔒 الأمان

- ✅ التخزين محلي فقط — لا بيانات تُرسل إلا للـ LLM
- ✅ التحقق من نوع الملف وحجمه قبل المعالجة
- ✅ تنظيف المدخلات
- ✅ تنظيف تلقائي للجلسات القديمة (كل ساعتين)
- ⚠️ لا ترفع ملف `.env` على GitHub

---

## 📚 التقنيات المستخدمة

| الطبقة | التقنية |
|--------|---------|
| Frontend | Gradio 4+ |
| LLM Orchestration | LangChain (LCEL) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Embedding Cache | LangChain `CacheBackedEmbeddings` |
| Vector DB | FAISS |
| LLM Provider | OpenRouter API |
| Document Loaders | PyPDF · Docx2txt · TextLoader |

---

## 🧪 تشغيل التقييم

```python
from app.evaluator import SystemEvaluator

eval = SystemEvaluator()

# اختبار إجابة
result = eval.evaluate_answer(
    question="ما هو موضوع المستند؟",
    answer=answer,
    expected_keywords=["موضوع", "مستند"],
    latency=2.1
)

# اختبار الأداء
perf = eval.latency_test(ask_question, "ما هو الموضوع؟", runs=5)
eval.print_summary()

# حفظ التقرير
eval.save_report("eval_report.json")
```

> 📝 يتم التقييم تلقائياً بعد كل سؤال حقيقي ويُحفظ في `eval_report.json`
