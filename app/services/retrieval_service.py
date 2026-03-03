from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS

from app.llm.llm_factory import get_llm

def build_casual_chain():
    """Simple chain for greetings and small talk — no retrieval."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly and helpful assistant.
                   Respond naturally and warmly to the user.
                   Keep your reply short and friendly."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnableLambda(lambda x: {
            "chat_history": x.get("chat_history", []),
            "question": x["question"],
        })
        | prompt
        | get_llm(streaming=True)
        | StrOutputParser()
    )
    return chain

_casual_chain = None

def get_casual_chain():
    global _casual_chain
    if _casual_chain is None:
        _casual_chain = build_casual_chain()
    return _casual_chain


def build_chain(vectorstore: FAISS):
    """
    LCEL chain that:
    1. Retrieves relevant docs via MMR
    2. Formats prompt
    3. Streams response token by token
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "lambda_mult": 0.7},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document assistant. You ONLY answer based on the document context provided below.

STRICT RULES — follow them without exception:
1. Use ONLY the information in the Document Context below to answer.
2. Do NOT use any outside knowledge, general knowledge, or training data.
3. If the answer is not found in the Document Context, respond with: "This information is not available in the uploaded document."
4. Never guess, infer, or make up information.
5. Mention page numbers when available (e.g., [Page 3]).

STRICT FORMATTING RULES — always apply:
- Use **bold** for key terms and important phrases.
- Use bullet points (- item) for lists.
- Use numbered lists (1. 2. 3.) for steps or ordered items.
- Use `## Heading` for sections when the answer has multiple parts.
- Use tables (| col | col |) when comparing items or presenting structured data.
- Keep paragraphs short and readable. Add a blank line between sections.
- Do NOT write long dense paragraphs — break them up.

Document Context:
{context}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs) -> str:
        parts = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            parts.append(f"[Page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def retrieve(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        return {
            **inputs,
            "context": format_docs(docs),
            "source_docs": docs,
        }

    def build_prompt_inputs(inputs: dict) -> dict:
        return {
            "context": inputs["context"],
            "chat_history": inputs.get("chat_history", []),
            "question": inputs["question"],
        }

    answer_chain = (
        RunnableLambda(build_prompt_inputs)
        | prompt
        | get_llm(streaming=True)
        | StrOutputParser()
    )

    chain = (
        RunnableLambda(retrieve)
        | {
            "answer":      answer_chain,
            "source_docs": RunnableLambda(lambda x: x["source_docs"]),
        }
    )

    return chain
