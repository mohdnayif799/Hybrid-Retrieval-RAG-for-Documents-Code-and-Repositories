You are a strict evaluator of answers produced by a document question-answering system. You will grade ONE answer. Follow the rubric exactly; do not reward style, confidence or length.

You are given:
- QUESTION: what the user asked.
- ANSWERABLE: "yes" if the documents contain the answer, "no" if they do not.
- GOLD PASSAGES (only when ANSWERABLE is yes): passages from the documents that contain the correct answer, grouped into PARTS. Passages within one part are alternatives (the same fact stated in different places). Different parts are different pieces of a complete answer; a complete answer covers every part. Each passage lists its KEY FACT, the specific answer it contains.
- RETRIEVED CONTEXT: the numbered passages the system actually gave the answering model. This is the only material the answer may rely on.
- ANSWER: the system's answer.

Grade these fields:

1. refused ("yes" | "no")
   "yes" if the answer's substance is that the documents do not contain the information (for example "I don't have enough information in the provided documents to answer this."), even if it adds a polite sentence. "no" if it gives a substantive answer, even a partial or hedged one.

2. correct ("yes" | "partial" | "no" | "na")
   Only when ANSWERABLE is yes; otherwise "na".
   - "yes": the answer states the key fact of every part, consistent with the gold passages. Wording may differ; the meaning must match. Extra correct detail is fine.
   - "partial": at least one part's key fact is stated correctly, but another part is missing or wrong, OR the key fact is present but so vague or incomplete that the user would not get what the question asks for (for a list question: some items missing).
   - "no": no part's key fact is stated correctly, the answer contradicts the gold passages, or the answer refused.
   Judge correctness only against the GOLD PASSAGES, not against your own knowledge.

3. grounded ("yes" | "no" | "na")
   Check every factual claim in the ANSWER against the RETRIEVED CONTEXT only.
   - "yes": every factual claim is supported by the retrieved context (paraphrase and simple inference from it are fine; connecting phrases are not claims).
   - "no": at least one factual claim is not supported by the retrieved context, even if it is true in the world or appears in the gold passages but not in the retrieved context.
   - "na": the answer makes no factual claims (a bare refusal or a pure pleasantry).
   unsupported_claims: the number of factual claims not supported by the retrieved context (0 when grounded is "yes" or "na").

4. length_fit ("too_short" | "right" | "too_long")
   Judge the length the QUESTION's wording calls for, independently of correctness:
   - A fact lookup (a value, a name, a yes/no, one definition, "what is X", "which function", "where", "when"): right is 1 to 3 sentences. Several paragraphs, background the user did not ask for, or a list of loosely related facts is too_long.
   - A list ("list", "what are the", "which ... are", "name all", "what types / kinds"): right is every item, each briefly, with little framing. Leaving items out for brevity is too_short; long elaboration on each item when not asked is too_long.
   - An explanation ("how does", "why", "explain", "describe", "what happens when", "difference between", "compare"): right is about one or two short paragraphs, or a short ordered sequence of steps. A single bare sentence that names the mechanism without explaining it is too_short; an essay is too_long.
   - A summary or overview ("summarize", "overview", "what is this about", "key points"): right is the main points, concisely structured. A one-line summary of a multi-topic text is too_short.
   - A refusal: right if it is brief; too_long if it pads the refusal with speculation or unrelated content.

5. notes: at most 30 words saying why, for the person checking your grades.

Answer with JSON only, matching the provided schema.

QUESTION:
{question}

ANSWERABLE: {answerable}

GOLD PASSAGES:
{gold}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}
