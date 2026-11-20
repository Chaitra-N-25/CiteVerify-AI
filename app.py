import os
import numpy as np

from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import fitz
import re
import requests
import time
from urllib.parse import quote
from typing import List, Dict, Tuple

# ============================================================
# CiteVerify AI - clean, cache-free baseline
# ============================================================

st.set_page_config(
    page_title="CiteVerify AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 CiteVerify AI")
st.caption("Scientific claim and citation checking — research decision support, not a substitute for expert review.")

st.info(
    "This version always reads the PDF currently selected in the uploader. "
    "It shows the exact filename and extracted DOI list so you can see which paper is being analysed."
)

DOI_PATTERN = re.compile(
    r"(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE
)

CLAIM_TERMS = (
    "we show", "results show", "we found", "we find", "demonstrate",
    "demonstrates", "demonstrated", "suggest", "suggests", "indicates",
    "indicate", "significantly", "improves", "increases", "reduces",
    "outperforms", "achieves", "propose", "proposed", "our results",
    "the results", "is associated with", "was associated with"
)

def extract_pdf_text(pdf_bytes: bytes) -> Tuple[str, int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n".join(pages), len(doc)

def clean_doi(raw: str) -> str:
    doi = raw.strip()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = doi.strip().rstrip(".,;:)]}>\"'")
    return doi

def find_dois(text: str) -> List[str]:
    found = []
    for match in DOI_PATTERN.findall(text):
        doi = clean_doi(match)
        if doi and doi.lower() not in [x.lower() for x in found]:
            found.append(doi)
    return found

def get_reference_section(text: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*(references|bibliography|literature cited)\s*:?\s*$", line, re.I):
            start = i + 1
            break
    if start is None:
        # Some PDFs have the heading embedded with surrounding text.
        for i, line in enumerate(lines):
            if re.search(r"\b(references|bibliography|literature cited)\b", line, re.I):
                start = i + 1
                break
    return "\n".join(lines[start:]) if start is not None else ""

def split_references(reference_text: str) -> List[Tuple[str, str]]:
    refs = []
    # Numbered reference entries such as [1], [2], 1., or 1)
    pattern = re.compile(r"(?m)^\s*(?:\[(\d{1,3})\]|(\d{1,3})[.)])\s+")
    matches = list(pattern.finditer(reference_text))
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(reference_text)
        number = match.group(1) or match.group(2)
        body = reference_text[match.end():end].strip()
        if body:
            refs.append((number, body))
    if not refs and reference_text.strip():
        # Keep a single unnumbered reference block instead of silently losing it.
        refs = [("?", x.strip()) for x in re.split(r"\n\s*\n", reference_text) if x.strip()]
    return refs

def extract_claims(text: str, limit: int = 35) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)

    claims = []

    for sentence in sentences:
        s = sentence.strip()

        if 45 <= len(s) <= 650 and any(term in s.lower() for term in CLAIM_TERMS):
            if s not in claims:
                claims.append(s)

        if len(claims) >= limit:
            break

    return claims


def citation_numbers(sentence: str) -> List[str]:
    nums = []

    for group in re.findall(r"\[([0-9,\-\s]+)\]", sentence):
        for part in group.split(","):
            part = part.strip()

            if re.fullmatch(r"\d+\s*-\s*\d+", part):
                start, end = map(int, part.split("-"))

                if end >= start and end - start <= 30:
                    nums.extend(str(n) for n in range(start, end + 1))

            elif part.isdigit():
                nums.append(part)

    return list(dict.fromkeys(nums))

def crossref_lookup(doi: str) -> Dict:
    url = "https://api.crossref.org/works/" + quote(doi, safe="")
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "CiteVerifyAI/1.0 (student project)"})
        if response.status_code == 200:
            item = response.json().get("message", {})
            title = (item.get("title") or ["Title unavailable"])[0]
            authors = item.get("author") or []
            author_text = ", ".join(
                " ".join(x for x in [a.get("given", ""), a.get("family", "")] if x)
                for a in authors[:8]
            ) or "Authors unavailable"
            year = "Year unavailable"
            date_parts = item.get("published-print", {}).get("date-parts") or item.get("published-online", {}).get("date-parts")
            if date_parts and date_parts[0]:
                year = str(date_parts[0][0])
            return {"ok": True, "status": response.status_code, "title": title, "authors": author_text, "year": year, "url": item.get("URL", "")}
        return {"ok": False, "status": response.status_code, "error": f"Crossref returned HTTP {response.status_code}"}
    except requests.RequestException as exc:
        return {"ok": False, "status": "network", "error": f"Network error: {exc}"}
    except (ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "status": "metadata", "error": f"Could not read Crossref response: {exc}"}

def openalex_pdf_url(doi: str):
    try:
        url = "https://api.openalex.org/works/https://doi.org/" + quote(doi, safe="")
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        best = data.get("best_oa_location") or {}
        return best.get("pdf_url") or (best.get("landing_page_url") if best.get("is_oa") else None)
    except Exception:
        return None

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        if len(s.strip()) > 35
    ]


def simple_relevance(claim: str, evidence: str) -> float:
    stop_words = {
        "the", "and", "for", "with", "that", "this",
        "from", "were", "was", "are", "our", "their",
        "into", "using", "which", "than", "they",
        "these", "can", "may", "could", "not"
    }

    claim_words = {
        w for w in re.findall(r"[a-z0-9]+", claim.lower())
        if len(w) > 2 and w not in stop_words
    }

    evidence_words = {
        w for w in re.findall(r"[a-z0-9]+", evidence.lower())
        if len(w) > 2 and w not in stop_words
    }

    if not claim_words:
        return 0.0

    return len(claim_words & evidence_words) / len(claim_words)


def best_evidence(claim: str, paper_text: str, top_n: int = 3):
    candidates = split_sentences(paper_text)

    scored = [
        (simple_relevance(claim, sentence), sentence)
        for sentence in candidates
    ]

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        (score, sentence)
        for score, sentence in scored[:top_n]
        if score > 0
    ]
# -------------------- Input --------------------
uploaded_main = st.file_uploader(
    "1. Upload the research paper you want to analyse (PDF)",
    type=["pdf"],
    key="main_pdf"
)

if uploaded_main is None:
    st.warning("Upload a research-paper PDF to begin.")
    st.stop()

main_bytes = uploaded_main.getvalue()
try:
    main_text, page_count = extract_pdf_text(main_bytes)
except Exception as exc:
    st.error(f"Could not read this PDF: {exc}")
    st.stop()

st.success(f"Loaded current file: {uploaded_main.name}")
c1, c2, c3 = st.columns(3)
c1.metric("PDF pages", page_count)
c2.metric("Extracted characters", f"{len(main_text):,}")
c3.metric("Unique DOI strings", len(find_dois(main_text)))

if len(main_text.strip()) < 100:
    st.error("Very little text was extracted. This may be a scanned PDF; OCR is not included in this version.")
    st.stop()
    # -------------------- Claims and references --------------------
st.divider()
st.header("2. Scientific claim extraction")
claims = extract_claims(main_text)
if claims:
    st.write(f"Found {len(claims)} possible claim sentences.")
    for i, claim in enumerate(claims, 1):
        st.markdown(f"**Claim {i}:** {claim}")
else:
    st.warning("No claim sentences matched the current claim-keyword rules. Try a text-based research PDF with a Results/Discussion section.")

st.divider()
st.header("3. Citation and reference extraction")
ref_text = get_reference_section(main_text)
references = split_references(ref_text) if ref_text else []
doi_to_reference = {}
if references:
    for number, ref in references:
        dois = find_dois(ref)
        for doi in dois:
            doi_to_reference[doi] = {"number": number, "text": ref}
    st.write(f"Detected {len(references)} reference entries.")
    st.write(f"Detected {len(doi_to_reference)} unique DOI(s) inside the reference list.")
    with st.expander("View extracted references"):
        for number, ref in references:
            st.markdown(f"**[{number}]** {ref}")
else:
    st.warning("Could not identify a References/Bibliography section. DOI strings found elsewhere in the PDF are shown separately and may not be cited references.")

all_dois = find_dois(ref_text) if ref_text else []
if not all_dois:
    all_dois = list(doi_to_reference.keys())

st.subheader("DOIs detected in the reference list")
if all_dois:
    for doi in all_dois:
        st.code(doi)
else:
    st.warning("No DOI was detected in the reference list. The app cannot retrieve cited papers by DOI; title-based lookup is not included in this version.")
    # -------------------- Crossref --------------------
st.divider()
st.header("4. Original research paper metadata retrieval")
metadata_by_doi = {}
if all_dois:
    max_dois = st.slider("Maximum number of DOI records to look up", 1, min(30, len(all_dois)), min(10, len(all_dois)))
    if st.button("Look up DOI metadata", type="primary"):
        progress = st.progress(0)
        for i, doi in enumerate(all_dois[:max_dois]):
            metadata_by_doi[doi] = crossref_lookup(doi)
            progress.progress((i + 1) / max_dois)
            time.sleep(0.05)
        st.session_state["cv_metadata"] = metadata_by_doi
    metadata_by_doi = st.session_state.get("cv_metadata", {})
    for doi in all_dois[:max_dois]:
        result = metadata_by_doi.get(doi)
        with st.expander(f"DOI: {doi}", expanded=(result is not None and not result.get("ok", False))):
            if result is None:
                st.caption("Not looked up yet. Click the button above.")
            elif result.get("ok"):
                st.success("Crossref metadata found.")
                st.write("**Title:**", result["title"])
                st.write("**Authors:**", result["authors"])
                st.write("**Year:**", result["year"])
                if result.get("url"):
                    st.write("**Publisher record:**", result["url"])
            else:
                st.error(result["error"])
                if result.get("status") == 404:
                    st.caption("This exact DOI was not found in Crossref. It may be incomplete, mistyped, or not registered with Crossref.")
            pdf_url = openalex_pdf_url(doi)
            if pdf_url:
                st.write("**Possible OpenAlex open-access PDF/landing page:**", pdf_url)
            else:
                st.caption("No open-access PDF link was found through OpenAlex. This does not mean the paper does not exist.")
                # -------------------- Original papers upload --------------------
st.divider()
st.header("5. Upload cited/original research PDFs")
st.write(
    "If a cited paper is paywalled or no open PDF link is found, download it from your library/publisher if you have access, "
    "then upload it here. You can select more than one PDF."
)
source_uploads = st.file_uploader(
    "Upload cited research paper PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
    key="source_pdfs"
)

source_texts = []
if source_uploads:
    for source_file in source_uploads:
        try:
            source_text, source_pages = extract_pdf_text(source_file.getvalue())
            source_texts.append((source_file.name, source_text))
            st.success(f"{source_file.name}: uploaded and text extracted ({source_pages} pages, {len(source_text):,} characters).")
        except Exception as exc:
            st.error(f"{source_file.name}: could not read PDF: {exc}")

# -------------------- Claim-citation mapping and evidence --------------------
st.divider()
st.header("6. Claim-to-citation mapping")
if claims:
    mapping_rows = []
    for idx, claim in enumerate(claims, 1):
        nums = citation_numbers(claim)
        mapping_rows.append((idx, claim, nums))
    for idx, claim, nums in mapping_rows:
        with st.expander(f"Claim {idx} — citation mapping"):
            st.write(claim)
            if nums:
                st.write("**Numbered citations detected:**", ", ".join(f"[{n}]" for n in nums))
                matched = [r for n, r in references if n in nums]
                if matched:
                    for ref in matched:
                        st.caption(ref)
                else:
                    st.warning("Citation number(s) were found, but no matching numbered reference entry was extracted.")
            else:
                st.caption("No numbered citation like [1] or [2, 3] was detected in this sentence.")
                # -------------------- Stage 7: Evidence Matching --------------------

st.divider()
st.header("7. Evidence Matching")

if not source_texts:
    st.warning("Upload at least one cited research paper PDF in Stage 5 to perform evidence matching.")

elif not claims:
    st.warning("No claims are available for evidence matching.")

else:
    evidence_results = []

    for idx, claim in enumerate(claims, 1):

        st.subheader(f"Claim {idx}")

        st.write(claim)

        combined_results = []

        for filename, paper_text in source_texts:

            matches = best_evidence(claim, paper_text, top_n=3)

            for score, sentence in matches:
                combined_results.append(
                    (score, filename, sentence)
                )

        combined_results.sort(
            key=lambda x: x[0],
            reverse=True
        )

        if combined_results:

            for score, filename, sentence in combined_results[:3]:

                st.write(f"**Source:** {filename}")

                st.write(f"**Relevance:** {score * 100:.1f}%")

                st.info(sentence)

            best_score = combined_results[0][0]

            evidence_results.append({
                "claim": claim,
                "score": best_score,
                "status": "Evidence found"
            })

        else:
            st.warning("No matching evidence found.")

            evidence_results.append({
                "claim": claim,
                "score": 0,
                "status": "No evidence found"
            })

    st.session_state["cv_evidence_results"] = evidence_results
    # -------------------- Stage 8: Verification Summary --------------------

st.divider()
st.header("8. Final Verification Summary")

results = st.session_state.get("cv_evidence_results", [])

st.subheader("Overall Evidence Relevance Score")

if results:
    overall_score = sum(
        item["score"] for item in results
    ) / len(results)

    st.metric(
        "Overall Score",
        f"{overall_score * 100:.1f}%"
    )

    st.progress(overall_score)

results = st.session_state.get("cv_evidence_results", [])

if not results:
    st.warning("Complete Stage 7 to generate the verification summary.")

else:
    total = len(results)

    supported = sum(
        1 for item in results if item["score"] >= 0.30
    )

    weak = sum(
        1 for item in results
        if 0 < item["score"] < 0.30
    )

    missing = sum(
        1 for item in results if item["score"] == 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Claims", total)
    col2.metric("Relevant Evidence", supported)
    col3.metric("Weak Evidence", weak)
    col4.metric("Missing Evidence", missing)

    st.subheader("Claim Verification Results")

    for index, item in enumerate(results, 1):

        with st.expander(f"Claim {index}"):

            st.write(item["claim"])

            st.write(
                f"Evidence relevance: {item['score'] * 100:.1f}%"
            )

            if item["score"] >= 0.30:
                st.success("Relevant evidence found")

            elif item["score"] > 0:
                st.warning("Weak evidence found")

            else:
                st.error("No matching evidence found")

    st.info(
        "Note: These results are based on text similarity. "
        "They do not prove that a scientific claim is true or false."
    )
# -------------------- Stage 9: AI Plagiarism Detection --------------------

st.divider()

st.header("9. AI Plagiarism Detection")

st.write(
    "Uses Gemini embeddings to identify potentially similar "
    "content between the uploaded research paper and comparison papers."
)

api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

if not api_key:

    st.error("Gemini API key not found. Set GEMINI_API_KEY in your terminal.")

elif not source_texts:

    st.warning(
        "Upload comparison research papers in Stage 5 first."
    )

else:

    if st.button("Run Gemini Plagiarism Detection", type="primary"):

        try:

            client = genai.Client(api_key=api_key)

            main_sentences = [
                s for s in split_sentences(main_text)
                if len(s) >= 70
            ][:40]

            source_sentences = []

            for filename, paper_text in source_texts:

                sentences = [
                    s for s in split_sentences(paper_text)
                    if len(s) >= 70
                ][:40]

                for sentence in sentences:
                    source_sentences.append((filename, sentence))

            if not main_sentences or not source_sentences:

                st.warning("Not enough text available for comparison.")

            else:

                all_texts = (
                    main_sentences +
                    [item[1] for item in source_sentences]
                )

                with st.spinner("Gemini is analysing text similarity..."):

                    response = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=all_texts,
                        config=types.EmbedContentConfig(
                            task_type="SEMANTIC_SIMILARITY"
                        )
                    )

                    vectors = np.array([
                        embedding.values
                        for embedding in response.embeddings
                    ])

                    main_vectors = vectors[:len(main_sentences)]

                    source_vectors = vectors[len(main_sentences):]

                    similarity_matrix = cosine_similarity(
                        main_vectors,
                        source_vectors
                    )

                results = []

                for i, sentence in enumerate(main_sentences):

                    best_index = int(
                        np.argmax(similarity_matrix[i])
                    )

                    score = float(
                        similarity_matrix[i][best_index]
                    )

                    filename, matched_sentence = source_sentences[
                        best_index
                    ]

                    results.append({
                        "original": sentence,
                        "matched": matched_sentence,
                        "source": filename,
                        "score": max(0, score) * 100
                    })

                results.sort(
                    key=lambda x: x["score"],
                    reverse=True
                )
                st.session_state["cv_plagiarism_results"] = results

                st.success("Plagiarism similarity analysis completed!")

                high = sum(
                    1 for r in results if r["score"] >= 85
                )

                moderate = sum(
                    1 for r in results
                    if 70 <= r["score"] < 85
                )

                low = sum(
                    1 for r in results if r["score"] < 70
                )

                c1, c2, c3 = st.columns(3)

                c1.metric("High Similarity", high)
                c2.metric("Moderate Similarity", moderate)
                c3.metric("Low Similarity", low)

                st.subheader("Potentially Similar Content")

                for index, item in enumerate(results, 1):

                    with st.expander(
                        f"Match {index} — {item['score']:.1f}% similarity"
                    ):

                        st.write("*Your paper:*")
                        st.write(item["original"])

                        st.write("*Matching source:*")
                        st.write(item["matched"])

                        st.write("*Source document:*")
                        st.write(item["source"])

                        st.progress(
                            min(item["score"] / 100, 1.0)
                        )

                st.info(
                    "Similarity scores identify potentially similar wording "
                    "or meaning. They do not prove plagiarism. "
                    "Human review is required."
                )

        except Exception as exc:

            st.error(f"Gemini API error: {exc}")
            # -------------------- Human Review --------------------

st.divider()
st.header("10. Human Review")

st.write(
    "Review the potentially similar content identified by Gemini "
    "and make the final decision."
)

review_results = st.session_state.get(
    "cv_plagiarism_results", []
)

if not review_results:
    st.warning("Run Gemini Plagiarism Detection first.")

else:
    if "cv_human_reviews" not in st.session_state:
        st.session_state["cv_human_reviews"] = {}

    for index, item in enumerate(review_results, 1):

        with st.expander(
            f"Review Match {index} — {item['score']:.1f}% similarity"
        ):

            st.write("*Original text:*")
            st.write(item["original"])

            st.write("*Matching text:*")
            st.write(item["matched"])

            st.write("*Source:*")
            st.write(item["source"])

            st.write("*Human Review Decision:*")

            decision = st.selectbox(
                "Select your decision",
                [
                    "Not reviewed",
                    "Possible plagiarism",
                    "Not plagiarism",
                    "Needs further investigation"
                ],
                key=f"human_decision_{index}"
            )

            comments = st.text_area(
                "Reviewer comments",
                key=f"human_comment_{index}"
            )

            st.session_state["cv_human_reviews"][index] = {
                "decision": decision,
                "comments": comments
            }

    st.subheader("Human Review Summary")

    reviewed = sum(
        1 for review in st.session_state["cv_human_reviews"].values()
        if review["decision"] != "Not reviewed"
    )

    st.metric("Matches Reviewed", f"{reviewed}/{len(review_results)}")

    st.success(
        "Human review decisions are saved during this Streamlit session."
    )