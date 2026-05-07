"""
Generate 50 fake judicial opinions for development and testing.
Saves output to data/raw/dummy_opinions.json.
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

COURTS = [
    "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7",
    "ca8", "ca9", "ca10", "ca11", "cadc", "cafc",
]

TOPICS = [
    "fourth_amendment",
    "due_process",
    "equal_protection",
    "first_amendment",
    "eighth_amendment",
    "fifth_amendment",
    "fourteenth_amendment",
]

# ── Text building blocks ──────────────────────────────────────────────────────

FOURTH_AMENDMENT_PARAGRAPHS = [
    (
        "The Fourth Amendment guarantees the right of the people to be secure in their persons, "
        "houses, papers, and effects against unreasonable searches and seizures. At the heart of "
        "this appeal is whether law enforcement's warrantless examination of the appellant's "
        "digital files fell within any recognized exception to the warrant requirement. We hold "
        "that it did not. The government's reliance on the automobile exception is misplaced; "
        "that doctrine extends to containers found in vehicles, not to the contents of a "
        "password-protected smartphone discovered incident to arrest."
    ),
    (
        "The touchstone of the Fourth Amendment is reasonableness. Terry v. Ohio, 392 U.S. 1, 9 "
        "(1968). We evaluate reasonableness by balancing the degree to which the challenged "
        "conduct intrudes upon an individual's privacy against the degree to which it is needed "
        "for the promotion of legitimate governmental interests. This balancing must account for "
        "the particular nature of cell phone data, which the Supreme Court has recognized "
        "implicates privacy concerns of an entirely different order than physical objects. "
        "Riley v. California, 573 U.S. 373, 393 (2014)."
    ),
    (
        "Probable cause to arrest does not, without more, furnish probable cause to search every "
        "digital repository the arrestee carries. The government argues that Officer Reyes's "
        "warrantless access to the GPS history on appellant's phone was justified by exigent "
        "circumstances. We are unpersuaded. The officers had ample time to apply for a warrant "
        "and faced no imminent destruction of evidence; the record reveals that the phone was "
        "seized and secured before any search occurred. Under these circumstances, the exigency "
        "exception cannot be stretched to sanction the search."
    ),
    (
        "We acknowledge the tension in our sister circuits on the question of whether the "
        "third-party doctrine of Smith v. Maryland, 442 U.S. 735 (1979), survives Carpenter "
        "v. United States, 585 U.S. 296 (2018), with respect to real-time location data "
        "obtained from a private carrier. The Sixth and Eighth Circuits have held that "
        "Carpenter is limited to historical cell-site location information spanning seven or "
        "more days. We disagree. Carpenter's logic rests on the comprehensiveness and "
        "intimacy of location data, qualities that do not vanish simply because the surveillance "
        "period is short."
    ),
    (
        "Suppression of evidence obtained through an unconstitutional search is the ordinary "
        "remedy. United States v. Leon, 468 U.S. 897, 906 (1984). The government invokes the "
        "good-faith exception, contending that the officers relied in objective good faith on "
        "binding circuit precedent that has since been abrogated. This argument has force where "
        "a prior controlling decision squarely authorized the conduct at issue. But no decision "
        "of this court or the Supreme Court specifically sanctioned a warrantless search of "
        "real-time GPS data obtained from a home security device. The good-faith exception "
        "therefore does not apply."
    ),
]

DUE_PROCESS_PARAGRAPHS = [
    (
        "The Due Process Clause of the Fourteenth Amendment prohibits the government from "
        "depriving any person of life, liberty, or property without due process of law. "
        "Procedural due process requires, at a minimum, notice and an opportunity to be heard "
        "at a meaningful time and in a meaningful manner. Mathews v. Eldridge, 424 U.S. 319, "
        "333 (1976). The three-part Mathews balancing test weighs: (1) the private interest "
        "affected; (2) the risk of erroneous deprivation and the probable value of additional "
        "safeguards; and (3) the government's interest, including fiscal and administrative "
        "burdens."
    ),
    (
        "Substantive due process protects individuals from certain government actions regardless "
        "of the fairness of the procedures used to implement them. Washington v. Glucksberg, "
        "521 U.S. 702, 720 (1997). Fundamental rights protected by substantive due process "
        "include those deeply rooted in this Nation's history and tradition and implicit in the "
        "concept of ordered liberty. The right asserted here -- the right to maintain a parental "
        "relationship without termination absent clear and convincing evidence of unfitness -- "
        "satisfies this demanding standard. Santosky v. Kramer, 455 U.S. 745, 769 (1982)."
    ),
    (
        "We apply the rational basis standard to plaintiff's substantive due process claim "
        "because the challenged regulation does not implicate a fundamental right or a suspect "
        "classification. Under rational basis review, a legislative classification must be "
        "upheld if it is rationally related to a legitimate governmental interest. The "
        "government's interest in protecting public health through mandatory vaccination "
        "requirements for healthcare workers is unquestionably legitimate, and the linkage "
        "between that requirement and the regulatory means chosen is neither arbitrary nor "
        "irrational."
    ),
    (
        "The doctrine of void for vagueness derives from the Due Process Clause and requires "
        "that penal statutes define criminal conduct with sufficient definiteness that ordinary "
        "people can understand what conduct is prohibited, and in a manner that does not "
        "encourage arbitrary and discriminatory enforcement. Kolender v. Lawson, 461 U.S. 352, "
        "357 (1983). The ordinance at issue here, which criminalizes 'loitering in a manner "
        "annoying to passers-by,' fails both prongs of this inquiry. It gives virtually no "
        "notice of what conduct is forbidden and vests police with unconstrained discretion."
    ),
    (
        "An agency's failure to follow its own procedural regulations may constitute a due "
        "process violation when the agency's internal rules are designed to afford the "
        "regulated party substantive protection. Morton v. Ruiz, 415 U.S. 199, 235 (1974). "
        "The Department concedes that it did not provide appellant with the individualized "
        "review promised in its published guidance before revoking her professional license. "
        "We find that this failure, combined with the severity of the deprivation -- loss of "
        "the ability to practice medicine -- rendered the deprivation constitutionally deficient."
    ),
]

EQUAL_PROTECTION_PARAGRAPHS = [
    (
        "The Equal Protection Clause of the Fourteenth Amendment commands that no state shall "
        "deny to any person within its jurisdiction the equal protection of the laws. "
        "Classifications based on race are subject to strict scrutiny: they must be narrowly "
        "tailored to serve a compelling governmental interest. Adarand Constructors, Inc. v. "
        "Pena, 515 U.S. 200, 227 (1995). The state has articulated a compelling interest in "
        "remedying the specific, identified effects of past discrimination in its public "
        "contracting programs, but the challenged set-aside program sweeps far more broadly "
        "than narrow tailoring permits."
    ),
    (
        "We review gender-based classifications under intermediate scrutiny, which requires "
        "that the classification serve important governmental objectives and that the "
        "discriminatory means be substantially related to those objectives. United States v. "
        "Virginia, 518 U.S. 515, 533 (1996). The government must show 'at least that the "
        "challenged classification serves important governmental objectives and that the "
        "discriminatory means employed are substantially related to the achievement of those "
        "objectives.' This standard is not toothless -- the justification must be genuine, "
        "not invented post hoc in response to litigation."
    ),
    (
        "Disparate impact alone does not prove discriminatory intent under the Equal Protection "
        "Clause. Washington v. Davis, 426 U.S. 229, 242 (1976). A plaintiff must demonstrate "
        "that the defendant acted with discriminatory purpose, which may be established through "
        "circumstantial evidence including the historical background of the decision, the "
        "sequence of events leading to the challenged action, and departures from normal "
        "procedural sequences. Village of Arlington Heights v. Metropolitan Housing Development "
        "Corp., 429 U.S. 252, 267-68 (1977)."
    ),
    (
        "The Supreme Court has held that laws imposing differential treatment on the basis of "
        "sexual orientation are subject to rational basis review. Romer v. Evans, 517 U.S. 620, "
        "631 (1996). Rational basis review is deferential, but it is not a rubber stamp. A "
        "classification that bears no rational relationship to a legitimate governmental purpose "
        "must fall, even if the government's interest is genuine. Here, the record discloses "
        "that the sole motivation for the challenged licensing restriction was 'bare desire to "
        "harm a politically unpopular group,' which can never constitute a legitimate "
        "governmental purpose."
    ),
    (
        "This court has recognized that facially neutral laws with racially disparate effects "
        "may violate the Equal Protection Clause when the plaintiff can point to a 'stark' "
        "pattern of discrimination unexplainable on grounds other than race. The statistical "
        "evidence presented here -- a six-to-one racial disparity in prosecutorial charging "
        "decisions persisting over a ten-year period -- combined with the absence of any "
        "race-neutral explanation in the record, is sufficient to survive summary judgment. "
        "We therefore remand for further proceedings on the selective prosecution claim."
    ),
]

FIRST_AMENDMENT_PARAGRAPHS = [
    (
        "The First Amendment prohibits the government from abridging the freedom of speech. "
        "Content-based restrictions on speech are presumptively unconstitutional and subject "
        "to strict scrutiny. Reed v. Town of Gilbert, 576 U.S. 155, 163 (2015). A law is "
        "content-based on its face if it applies to particular speech because of the topic "
        "discussed or the idea or message expressed. The ordinance challenged here "
        "differentiates between 'political' and 'commercial' signage based entirely on the "
        "content of the message and therefore triggers strict scrutiny."
    ),
    (
        "Prior restraints on speech bear a heavy presumption against their constitutional "
        "validity. A government order that prohibits speech before it is communicated must "
        "be supported by evidence that publication will cause direct, immediate, and irreparable "
        "harm of the highest order. New York Times Co. v. United States, 403 U.S. 713, 714 "
        "(1971) (per curiam). The district court's injunction restraining publication of "
        "the whistleblower's account cannot survive this exacting standard on the record "
        "before us."
    ),
    (
        "Government speech is not subject to First Amendment scrutiny; when the government "
        "speaks, it is entitled to promote a program, espouse a policy, or take positions "
        "without providing equal time to dissenting views. Matal v. Tam, 582 U.S. 218, 235 "
        "(2017). But the line between government speech and private speech that the government "
        "has chosen to regulate is not always clear. The specialty license plate program at "
        "issue here -- in which the state designs the plates, controls the content, and "
        "ultimately authorizes all messages -- is more properly characterized as government "
        "speech than private speech on a government platform."
    ),
]

EIGHTH_AMENDMENT_PARAGRAPHS = [
    (
        "The Eighth Amendment's prohibition on cruel and unusual punishment draws its meaning "
        "from the evolving standards of decency that mark the progress of a maturing society. "
        "Trop v. Dulles, 356 U.S. 86, 101 (1958). In assessing whether a punishment is grossly "
        "disproportionate, we look to objective indicators of society's standards, including "
        "legislative enactments and actual sentencing practices. Graham v. Florida, 560 U.S. "
        "48, 61 (2010). A life sentence without the possibility of parole imposed on a "
        "non-homicide juvenile offender categorically violates the Eighth Amendment."
    ),
    (
        "Deliberate indifference to serious medical needs of prisoners constitutes the "
        "unnecessary and wanton infliction of pain proscribed by the Eighth Amendment. "
        "Estelle v. Gamble, 429 U.S. 97, 104 (1976). The deliberate indifference standard "
        "has both an objective and a subjective component: the deprivation must be "
        "sufficiently serious, and the defendant must have acted with the requisite "
        "culpable state of mind. The plaintiff's undisputed evidence that correctional "
        "officers disregarded his repeated requests for insulin for seventy-two hours "
        "satisfies both components."
    ),
]

FIFTH_AMENDMENT_PARAGRAPHS = [
    (
        "The Takings Clause of the Fifth Amendment provides that private property shall not "
        "be taken for public use without just compensation. A per se taking occurs when the "
        "government physically appropriates the property of another, even temporarily. Cedar "
        "Point Nursery v. Hassid, 594 U.S. 139, 147-48 (2021). The regulation challenged "
        "here requires property owners to grant union organizers access to their private "
        "agricultural land for three hours per day, two hundred twenty days per year. This "
        "constitutes a physical appropriation of the property owners' right to exclude -- "
        "one of the most essential sticks in the bundle of property rights -- and is a "
        "per se taking requiring just compensation."
    ),
    (
        "The Fifth Amendment's Self-Incrimination Clause prohibits the government from "
        "compelling a person to be a witness against himself in a criminal case. The "
        "privilege protects against any disclosures that the witness reasonably believes "
        "could be used in a criminal prosecution or could lead to other evidence that "
        "might be so used. Kastigar v. United States, 406 U.S. 441, 444-45 (1972). "
        "The act of producing documents can itself be testimonial if the act concedes the "
        "existence, authenticity, or possession of the documents. Fisher v. United States, "
        "425 U.S. 391, 410 (1976)."
    ),
]

TOPIC_PARAGRAPHS = {
    "fourth_amendment": FOURTH_AMENDMENT_PARAGRAPHS,
    "due_process": DUE_PROCESS_PARAGRAPHS,
    "equal_protection": EQUAL_PROTECTION_PARAGRAPHS,
    "first_amendment": FIRST_AMENDMENT_PARAGRAPHS,
    "eighth_amendment": EIGHTH_AMENDMENT_PARAGRAPHS,
    "fifth_amendment": FIFTH_AMENDMENT_PARAGRAPHS,
    "fourteenth_amendment": DUE_PROCESS_PARAGRAPHS + EQUAL_PROTECTION_PARAGRAPHS,
}

DISSENT_OPENINGS = [
    "I respectfully dissent. The majority's analysis, while carefully reasoned, reaches a conclusion that I believe departs from controlling precedent in a manner that will create significant confusion in the lower courts.",
    "I dissent. The majority today expands the exclusionary rule in a way that the Supreme Court has never sanctioned and that finds no support in the text or history of the Fourth Amendment.",
    "With respect, I cannot join the majority's opinion. The constitutional question presented here is one on which reasonable jurists disagree, but the balance of authority favors the government's position.",
    "I write separately to dissent from the majority's holding. In my view, the majority's reliance on Carpenter v. United States overreads that decision and ignores the significant limiting principles articulated therein.",
    "The majority reaches the right result for the wrong reasons. I would affirm on the narrower ground that the appellant failed to preserve the constitutional objection below, and therefore write separately.",
]

CONCURRENCE_OPENINGS = [
    "I concur in the judgment, but write separately to note that the majority's reasoning is broader than necessary to resolve this appeal.",
    "I join the majority opinion in full but write separately to emphasize what I view as the critical limiting principle of today's decision.",
    "I concur in the result. I agree that the judgment below must be reversed, but I reach that conclusion through a different analytical path than the majority.",
    "Although I join the majority's disposition, I am troubled by its dictum regarding the application of the third-party doctrine. Those observations are unnecessary to the holding and may generate mischief in future cases.",
]

CLOSING_SENTENCES = [
    "For the foregoing reasons, we AFFIRM the judgment of the district court.",
    "For the reasons stated above, we REVERSE and REMAND for proceedings consistent with this opinion.",
    "Accordingly, the judgment of the district court is AFFIRMED in part, REVERSED in part, and REMANDED.",
    "We therefore VACATE the sentence and REMAND for resentencing.",
    "The order of the district court granting summary judgment is REVERSED.",
    "We AFFIRM the district court's denial of the motion to suppress.",
    "For these reasons, we GRANT the petition for rehearing en banc and REVERSE the panel decision.",
]


def random_date(start_year: int = 2010, end_year: int = 2020) -> str:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


def build_plain_text(topic: str, opinion_type: str) -> str:
    pool = TOPIC_PARAGRAPHS.get(topic, DUE_PROCESS_PARAGRAPHS)
    n_body = random.randint(3, 5)
    body_paras = random.sample(pool, min(n_body, len(pool)))

    paragraphs = []

    if opinion_type == "dissent":
        paragraphs.append(random.choice(DISSENT_OPENINGS))
    elif opinion_type == "concurrence":
        paragraphs.append(random.choice(CONCURRENCE_OPENINGS))

    paragraphs.extend(body_paras)
    paragraphs.append(random.choice(CLOSING_SENTENCES))

    return "\n\n".join(paragraphs)


def build_cluster_layout(n_opinions: int = 50):
    """
    Return a list of (opinion_type, cluster_id) tuples summing to n_opinions.
    Clusters: majority-only, majority+dissent, majority+concurrence.
    """
    layout = []
    cluster_counter = 1

    # majority + dissent pairs (10 clusters = 20 opinions)
    n_pairs = 10
    for _ in range(n_pairs):
        cid = f"cl_{cluster_counter:03d}"
        cluster_counter += 1
        layout.append(("majority", cid))
        layout.append(("dissent", cid))

    # majority + concurrence pairs (5 clusters = 10 opinions)
    n_concur = 5
    for _ in range(n_concur):
        cid = f"cl_{cluster_counter:03d}"
        cluster_counter += 1
        layout.append(("majority", cid))
        layout.append(("concurrence", cid))

    # majority-only (20 clusters = 20 opinions)
    n_solo = n_opinions - (n_pairs * 2) - (n_concur * 2)
    for _ in range(n_solo):
        cid = f"cl_{cluster_counter:03d}"
        cluster_counter += 1
        layout.append(("majority", cid))

    random.shuffle(layout)
    return layout


def generate_opinions(n: int = 50) -> list[dict]:
    layout = build_cluster_layout(n)
    opinions = []

    for idx, (op_type, cluster_id) in enumerate(layout, start=1):
        topic = random.choice(TOPICS)
        opinion = {
            "id": f"op_{idx:03d}",
            "date_filed": random_date(),
            "court": random.choice(COURTS),
            "type": op_type,
            "cluster_id": cluster_id,
            "plain_text": build_plain_text(topic, op_type),
        }
        opinions.append(opinion)

    return opinions


def main():
    out_dir = Path(__file__).parent / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dummy_opinions.json"

    opinions = generate_opinions(50)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(opinions, f, indent=2, ensure_ascii=True)

    print(f"Wrote {len(opinions)} opinions to {out_path}")

    type_counts = {}
    for op in opinions:
        type_counts[op["type"]] = type_counts.get(op["type"], 0) + 1
    print(f"Type breakdown: {type_counts}")

    cluster_ids = {op["cluster_id"] for op in opinions}
    print(f"Unique clusters: {len(cluster_ids)}")


if __name__ == "__main__":
    main()
