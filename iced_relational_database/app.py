from __future__ import annotations

import sqlite3
import re
import json
from collections import Counter
from itertools import combinations
from math import cos, pi, sin
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "iced_relational_database" / "iced_relational.sqlite"

st.set_page_config(
    page_title="ICED — Topic and maturity explorer",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE_PATH}")
    return sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True, check_same_thread=False)


@st.cache_data
def read_query(query: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, get_connection(), params=params)


@st.cache_data
def load_topics() -> pd.DataFrame:
    result = read_query(
        "SELECT topic_id, topic_label FROM topics WHERE is_outlier = 0 ORDER BY topic_label"
    )
    result["topic_name"] = result.apply(
        lambda row: format_topic_name(row["topic_label"], row["topic_id"]),
        axis=1,
    )
    return result


def format_topic_name(value: object, topic_id: object = None) -> str:
    """Create a stable, traceable label from the representative topic terms."""
    raw = re.sub(r"^(?:Topic\s*)?T?\d+[_\s]*", "", str(value), flags=re.IGNORECASE)
    terms = [term.replace("-", " ").strip().lower() for term in re.split(r"[_\s]+", raw) if term.strip()]
    stopwords = {"topic", "none", "nan", "and", "the", "of", "for", "with"}
    terms = list(dict.fromkeys(term for term in terms if term not in stopwords))
    keywords = " · ".join(terms[:4]) if terms else "unlabelled"
    stable_id = f"T{int(topic_id):02d}" if pd.notna(topic_id) else "T?"
    return f"{stable_id} — {keywords}"


def choose_document_title(
    title_clean: object, metadata: object, fallback_title: object
) -> str:
    """Prefer a cleaned title, then metadata title fields, then the legacy title."""
    if isinstance(title_clean, str) and title_clean.strip():
        return title_clean.strip()
    if isinstance(metadata, str):
        try:
            metadata_value = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_value = {}
        if isinstance(metadata_value, dict):
            for key in ("title_clean", "title", "document_title"):
                value = metadata_value.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return str(fallback_title or "Untitled document").strip()


@st.cache_data
def load_sources() -> list[str]:
    return read_query("SELECT source_code FROM sources ORDER BY source_code")["source_code"].tolist()


@st.cache_data
def load_years() -> list[int]:
    return read_query(
        "SELECT DISTINCT publication_year FROM documents "
        "WHERE publication_year IS NOT NULL ORDER BY publication_year"
    )["publication_year"].astype(int).tolist()


def build_filters(
    topic_ids: list[int],
    sources: list[str],
    years: list[int],
    trls: list[int],
) -> tuple[str, list]:
    clauses = ["dv.topic_id IS NOT NULL", "dv.topic_id != -1"]
    params: list = []
    if topic_ids:
        clauses.append("dv.topic_id IN (" + ",".join("?" for _ in topic_ids) + ")")
        params.extend(topic_ids)
    if sources:
        clauses.append("dv.source_code IN (" + ",".join("?" for _ in sources) + ")")
        params.extend(sources)
    if years:
        clauses.append("dv.publication_year IN (" + ",".join("?" for _ in years) + ")")
        params.extend(years)
    if trls:
        clauses.append("dv.predicted_trl IN (" + ",".join("?" for _ in trls) + ")")
        params.extend(trls)
    return " AND ".join(clauses), params


def filtered_documents(
    topic_ids: list[int],
    sources: list[str],
    years: list[int],
    trls: list[int],
) -> pd.DataFrame:
    where, params = build_filters(topic_ids, sources, years, trls)
    documents = read_query(
        f"""
        SELECT DISTINCT dv.document_id, dv.source_code, dv.publication_year,
               dv.predicted_trl, dv.trl_label, dv.trl_similarity,
               dv.topic_id, dv.topic_label, dv.topic_score, d.title,
               d.title_clean, d.metadata, d.keywords,
               COALESCE(d.pretreated_text, d.abstract, d.title, '') AS document_text
        FROM document_visualization dv
        JOIN documents d ON d.document_id = dv.document_id
        WHERE {where}
        ORDER BY dv.publication_year, dv.document_id
        """,
        tuple(params),
    )
    topic_names = load_topics().set_index("topic_id")["topic_name"]
    documents["topic_name"] = documents["topic_id"].map(topic_names)
    documents["display_title"] = documents.apply(
        lambda row: choose_document_title(
            row["title_clean"], row["metadata"], row["title"]
        ),
        axis=1,
    )
    return documents


def topic_distribution(documents: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        documents.groupby(["topic_id", "topic_name"], as_index=False)["document_id"]
        .nunique()
        .rename(columns={"document_id": "documents"})
        .sort_values("documents", ascending=False)
    )
    distribution["topic_label"] = distribution["topic_name"]
    return distribution


def render_topic_explorer(documents: pd.DataFrame) -> None:
    st.subheader("Topic explorer")
    if documents.empty:
        st.info("No documents match the current filters.")
        return
    topics = topic_distribution(documents)
    selected = st.selectbox(
        "Select a topic to inspect",
        topics["topic_id"].tolist(),
        format_func=lambda value: topics.loc[topics.topic_id.eq(value), "topic_name"].iloc[0],
    )
    selected_documents = documents[documents["topic_id"].eq(selected)].copy()
    st.caption("Cleaned titles are shown when available; click a column header to sort the table.")
    st.dataframe(
        selected_documents[
            [
                "document_id",
                "source_code",
                "publication_year",
                "predicted_trl",
                "trl_similarity",
                "display_title",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download selected topic documents",
        selected_documents.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"topic_{selected}_documents.csv",
        mime="text/csv",
    )


def render_dashboard(documents: pd.DataFrame, topic_view: str) -> None:
    st.subheader("D1 — Topic landscape")
    distribution = topic_distribution(documents)
    if distribution.empty:
        st.info("No documents match the current filters.")
        return
    fig = px.bar(
        distribution.sort_values("documents"),
        x="documents",
        y="topic_label",
        orientation="h",
        labels={"documents": "Distinct documents", "topic_label": "Topic"},
    )
    st.plotly_chart(fig, use_container_width=True, key="dashboard_topic_distribution")

    left, right = st.columns(2)
    with left:
        st.subheader("Topic × Source")
        source = (
            documents.groupby(["topic_name", "source_code"], as_index=False)["document_id"]
            .nunique()
            .rename(columns={"topic_name": "topic_label", "document_id": "documents"})
        )
        heat = source.pivot(index="topic_label", columns="source_code", values="documents").fillna(0)
        if topic_view == "Separate figure per topic" and len(heat.index) > 1:
            for topic in heat.index:
                st.plotly_chart(
                    px.bar(
                        source[source["topic_label"].eq(topic)],
                        x="source_code",
                        y="documents",
                        title=topic,
                        labels={"source_code": "Source", "documents": "Documents"},
                    ),
                    use_container_width=True,
                    key=f"dashboard_source_{topic}",
                )
        else:
            st.plotly_chart(
                px.imshow(
                    heat,
                    text_auto=True,
                    aspect="auto",
                    labels={"x": "Source", "y": "Topic", "color": "Documents"},
                ),
                use_container_width=True,
                key="dashboard_topic_source",
            )
    with right:
        st.subheader("Topic × Year")
        temporal = (
            documents.groupby(["topic_name", "publication_year"], as_index=False)["document_id"]
            .nunique()
            .rename(columns={"topic_name": "topic_label", "document_id": "documents"})
        )
        if topic_view == "Separate figure per topic" and len(temporal["topic_label"].unique()) > 1:
            for topic in temporal["topic_label"].drop_duplicates():
                st.plotly_chart(
                    px.line(
                        temporal[temporal["topic_label"].eq(topic)],
                        x="publication_year",
                        y="documents",
                        markers=True,
                        title=topic,
                        labels={"documents": "Distinct documents", "publication_year": "Year"},
                    ),
                    use_container_width=True,
                    key=f"dashboard_year_{topic}",
                )
        else:
            st.plotly_chart(
                px.line(
                    temporal,
                    x="publication_year",
                    y="documents",
                    color="topic_label" if topic_view == "Different colors" else None,
                    markers=True,
                    labels={"documents": "Distinct documents", "publication_year": "Year"},
                ),
                use_container_width=True,
                key="dashboard_topic_year",
            )

    st.subheader("D4 — Topic × estimated TRL")
    maturity = (
        documents.groupby(["topic_name", "predicted_trl"], as_index=False)["document_id"]
        .nunique()
        .rename(columns={"topic_name": "topic_label", "document_id": "documents"})
    )
    maturity_heat = maturity.pivot(index="topic_label", columns="predicted_trl", values="documents").fillna(0)
    if topic_view == "Separate figure per topic" and len(maturity_heat.index) > 1:
        for topic in maturity_heat.index:
            topic_maturity = maturity[maturity["topic_label"].eq(topic)]
            st.plotly_chart(
                px.bar(
                    topic_maturity,
                    x="predicted_trl",
                    y="documents",
                    title=topic,
                    labels={"predicted_trl": "Estimated TRL", "documents": "Documents"},
                ),
                use_container_width=True,
                key=f"dashboard_maturity_{topic}",
            )
    else:
        st.plotly_chart(
            px.imshow(
                maturity_heat,
                text_auto=True,
                aspect="auto",
                labels={"x": "Estimated TRL", "y": "Topic", "color": "Documents"},
                color_continuous_scale="Cividis",
            ),
            use_container_width=True,
            key="dashboard_topic_maturity",
        )


def render_relationships(topic_ids: list[int], key_prefix: str = "relationships") -> None:
    st.subheader("D3 — Shared-document relationships")
    if len(topic_ids) < 2:
        st.info("Select at least two topics in the sidebar to display relationships.")
        return
    placeholders = ",".join("?" for _ in topic_ids)
    relations = read_query(
        f"""
        SELECT r.topic_a, r.topic_b, r.shared_documents,
               a.topic_label AS topic_a_label, b.topic_label AS topic_b_label
        FROM topic_relations r
        JOIN topics a ON a.topic_id = r.topic_a
        JOIN topics b ON b.topic_id = r.topic_b
        WHERE r.topic_a IN ({placeholders}) AND r.topic_b IN ({placeholders})
        ORDER BY r.shared_documents DESC
        """,
        tuple(topic_ids + topic_ids),
    )
    relations["topic_a_name"] = relations.apply(
        lambda row: format_topic_name(row["topic_a_label"], row["topic_a"]), axis=1
    )
    relations["topic_b_name"] = relations.apply(
        lambda row: format_topic_name(row["topic_b_label"], row["topic_b"]), axis=1
    )
    st.dataframe(
        relations[["topic_a_name", "topic_b_name", "shared_documents"]].rename(
            columns={
                "topic_a_name": "Topic A",
                "topic_b_name": "Topic B",
                "shared_documents": "Shared documents",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    matrix = relations.pivot(
        index="topic_a_name", columns="topic_b_name", values="shared_documents"
    ).replace(0, float("nan"))
    if not matrix.empty:
        st.plotly_chart(
            px.imshow(
                matrix,
                text_auto=True,
                aspect="auto",
                labels={"x": "Topic B", "y": "Topic A", "color": "Shared documents"},
                color_continuous_scale="Cividis",
            ),
            use_container_width=True,
            key=f"{key_prefix}_shared_documents_matrix",
        )


def build_semantic_edges(
    documents: pd.DataFrame, topic_ids: list[int]
) -> pd.DataFrame:
    """Build topic edges from cosine similarity between TF-IDF topic centroids."""
    corpus = (
        documents[documents["topic_id"].isin(topic_ids)]
        .drop_duplicates(["topic_id", "document_id"])
        .assign(document_text=lambda frame: frame["document_text"].fillna(""))
        .groupby(["topic_id", "topic_name"], as_index=False)["document_text"]
        .agg(" ".join)
    )
    corpus = corpus[corpus["document_text"].str.strip().ne("")]
    if len(corpus) < 2:
        return pd.DataFrame(
            columns=["topic_a_name", "topic_b_name", "cosine_similarity"]
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus["document_text"])
    similarities = cosine_similarity(matrix)
    rows = []
    for left, right in combinations(range(len(corpus)), 2):
        rows.append(
            {
                "topic_a_name": corpus.iloc[left]["topic_name"],
                "topic_b_name": corpus.iloc[right]["topic_name"],
                "cosine_similarity": float(similarities[left, right]),
            }
        )
    return pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)


def render_graph_section(documents: pd.DataFrame, topic_ids: list[int]) -> None:
    st.subheader("Relational and semantic graphs")
    if len(topic_ids) < 2:
        st.info("Select at least two topics in the sidebar to display graphs.")
        return

    placeholders = ",".join("?" for _ in topic_ids)
    relations = read_query(
        f"""
        SELECT r.topic_a, r.topic_b, r.shared_documents, r.lift, r.phi,
               r.fisher_q_bh, a.topic_label AS topic_a_label,
               b.topic_label AS topic_b_label
        FROM topic_relations r
        JOIN topics a ON a.topic_id = r.topic_a
        JOIN topics b ON b.topic_id = r.topic_b
        WHERE r.topic_a IN ({placeholders}) AND r.topic_b IN ({placeholders})
        """,
        tuple(topic_ids + topic_ids),
    )
    relations["topic_a_name"] = relations.apply(
        lambda row: format_topic_name(row["topic_a_label"], row["topic_a"]), axis=1
    )
    relations["topic_b_name"] = relations.apply(
        lambda row: format_topic_name(row["topic_b_label"], row["topic_b"]), axis=1
    )
    graph_tab_relational, graph_tab_semantic = st.tabs(
        ["Relational graph", "Semantic graph"]
    )

    def graph_figure(edges: pd.DataFrame, weight_column: str, title: str) -> go.Figure:
        labels = pd.unique(edges[["topic_a_name", "topic_b_name"]].values.ravel("K")).tolist()
        graph = nx.Graph()
        graph.add_nodes_from(labels)
        graph.add_edges_from(
            (row.topic_a_name, row.topic_b_name)
            for row in edges.itertuples(index=False)
        )
        positions = nx.spring_layout(graph, seed=42, weight=None)
        edge_x: list[float] = []
        edge_y: list[float] = []
        for row in edges.itertuples(index=False):
            x1, y1 = positions[row.topic_a_name]
            x2, y2 = positions[row.topic_b_name]
            edge_x.extend([x1, x2, None])
            edge_y.extend([y1, y2, None])
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1.5, color="#7F7F7F"),
                hoverinfo="skip",
                name="Topic association",
            )
        )
        node_x = [positions[label][0] for label in labels]
        node_y = [positions[label][1] for label in labels]
        degree = {label: 0.0 for label in labels}
        for row in edges.itertuples(index=False):
            weight = abs(float(getattr(row, weight_column)))
            degree[row.topic_a_name] += weight
            degree[row.topic_b_name] += weight
        figure.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=[label[:28] for label in labels],
                textposition="middle center",
                customdata=[[label, round(degree[label], 3)] for label in labels],
                hovertemplate="Topic: %{customdata[0]}<br>Weighted degree: %{customdata[1]}<extra></extra>",
                marker=dict(
                    size=[22 + 8 * degree[label] / max(degree.values(), default=1) for label in labels],
                    color="#0072B2",
                    line=dict(width=1, color="white"),
                ),
                name="Topic",
            )
        )
        figure.update_layout(
            title=title,
            showlegend=False,
            height=650,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=55, b=10),
        )
        figure.add_annotation(
            text="Node size = weighted degree · edge width = association strength",
            x=0,
            y=-0.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=11, color="#555555"),
        )
        return figure

    with graph_tab_relational:
        relational = relations[relations["shared_documents"] > 0].copy()
        st.caption("Nodes are topics; edges count documents assigned to both topics. The table is sorted by shared-document count.")
        max_edges = max(len(relational), 1)
        edge_limit = st.slider(
            "Maximum relational edges",
            min_value=1,
            max_value=max_edges,
            value=min(25, max_edges),
            key="relational_edge_limit",
        )
        relational = relational.sort_values("shared_documents", ascending=False).head(edge_limit)
        st.dataframe(
            relational[
                ["topic_a_name", "topic_b_name", "shared_documents"]
            ].sort_values("shared_documents", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        if not relational.empty:
            st.plotly_chart(
                graph_figure(
                    relational,
                    "shared_documents",
                    "Relational graph — shared documents",
                ),
                use_container_width=True,
                key="graph_relational_shared_documents",
            )

    with graph_tab_semantic:
        semantic_all = build_semantic_edges(documents, topic_ids)
        threshold = st.slider(
            "Minimum cosine similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            key="semantic_cosine_threshold",
        )
        top_k = st.slider(
            "Maximum semantic neighbours per topic",
            min_value=1,
            max_value=max(len(topic_ids) - 1, 1),
            value=min(3, max(len(topic_ids) - 1, 1)),
            key="semantic_top_k",
        )
        candidate_edges = pd.concat(
            [
                semantic_all.assign(
                    source_topic=semantic_all["topic_a_name"],
                    neighbour=semantic_all["topic_b_name"],
                ),
                semantic_all.assign(
                    source_topic=semantic_all["topic_b_name"],
                    neighbour=semantic_all["topic_a_name"],
                ),
            ],
            ignore_index=True,
        )
        candidate_edges = candidate_edges.sort_values(
            ["source_topic", "cosine_similarity"], ascending=[True, False]
        )
        candidate_edges = candidate_edges.groupby("source_topic", as_index=False).head(top_k)
        semantic = candidate_edges[
            candidate_edges["cosine_similarity"] >= threshold
        ].copy()
        semantic["edge_key"] = semantic.apply(
            lambda row: tuple(sorted((row["topic_a_name"], row["topic_b_name"]))),
            axis=1,
        )
        semantic = semantic.drop_duplicates("edge_key").drop(columns="edge_key")
        st.caption("Nodes are topics; edges connect topics whose TF-IDF text centroids exceed the selected cosine-similarity threshold. The table is sorted by similarity.")
        st.dataframe(
            semantic[
                [
                    "topic_a_name",
                    "topic_b_name",
                    "cosine_similarity",
                ]
            ].sort_values("cosine_similarity", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        if not semantic.empty:
            st.plotly_chart(
                graph_figure(
                    semantic,
                    "cosine_similarity",
                    "Semantic graph — TF-IDF cosine similarity",
                ),
                use_container_width=True,
                key="graph_semantic_associations",
            )
        else:
            st.info("No semantic edges meet the selected cosine-similarity threshold.")


def render_integrated_view(documents: pd.DataFrame) -> None:
    st.subheader("Integrated view — Time × Topic × Source × TRL")
    integrated = (
        documents.groupby(
            ["publication_year", "topic_name", "source_code", "predicted_trl"], as_index=False
        )["document_id"]
        .nunique()
        .rename(columns={"topic_name": "topic_label", "document_id": "documents"})
        .dropna(subset=["publication_year", "predicted_trl"])
    )
    st.plotly_chart(
        px.scatter(
            integrated,
            x="publication_year",
            y="predicted_trl",
            size="documents",
            color="source_code",
            hover_data=["topic_label", "documents"],
            labels={"publication_year": "Year", "predicted_trl": "Estimated TRL"},
        ),
        use_container_width=True,
        key="integrated_time_topic_source_trl",
    )


def render_additional_views(documents: pd.DataFrame) -> None:
    st.subheader("Additional visualizations")
    if documents.empty:
        st.info("No documents match the current filters.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("#### Documentary activity by source and year")
        source_year = (
            documents.groupby(["publication_year", "source_code"], as_index=False)["document_id"]
            .nunique()
            .rename(columns={"document_id": "documents"})
        )
        st.plotly_chart(
            px.bar(
                source_year,
                x="publication_year",
                y="documents",
                color="source_code",
                barmode="stack",
                labels={"publication_year": "Year", "documents": "Distinct documents"},
            ),
            use_container_width=True,
            key="additional_source_year",
        )
    with right:
        st.markdown("#### Estimated TRL distribution")
        trl_distribution = (
            documents.groupby(["source_code", "predicted_trl"], as_index=False)["document_id"]
            .nunique()
            .rename(columns={"document_id": "documents"})
        )
        st.plotly_chart(
            px.bar(
                trl_distribution,
                x="predicted_trl",
                y="documents",
                color="source_code",
                barmode="group",
                labels={"predicted_trl": "Estimated TRL", "documents": "Distinct documents"},
            ),
            use_container_width=True,
            key="additional_trl_distribution",
        )

    st.markdown("#### Topic score distribution")
    score_data = documents.dropna(subset=["topic_score"]).copy()
    if not score_data.empty:
        score_data["topic_label"] = score_data["topic_name"]
        st.plotly_chart(
            px.box(
                score_data,
                x="topic_label",
                y="topic_score",
                color="source_code",
                points="outliers",
                labels={"topic_score": "Topic score", "topic_label": "Topic"},
            ).update_layout(xaxis_tickangle=-45),
            use_container_width=True,
            key="additional_topic_score",
        )

    st.markdown("#### Topic coverage overview")
    coverage = topic_distribution(documents)
    coverage["topic"] = coverage["topic_label"].str.slice(0, 45)
    st.plotly_chart(
        px.treemap(
            coverage,
            path=["topic"],
            values="documents",
            color="documents",
            color_continuous_scale="Blues",
            labels={"documents": "Distinct documents"},
        ),
        use_container_width=True,
        key="additional_topic_coverage",
    )


def render_legacy_views(documents: pd.DataFrame, topic_ids: list[int]) -> None:
    """Reproduce the main exploratory plots from the earlier notebooks."""
    st.subheader("Historical exploratory visualizations")
    if documents.empty:
        st.info("No documents match the current filters.")
        return

    st.markdown("#### Topic × time × estimated TRL")
    temporal = documents.dropna(
        subset=["publication_year", "predicted_trl", "topic_name"]
    ).copy()
    if not temporal.empty:
        temporal["topic_score"] = temporal["topic_score"].fillna(0.12)
        temporal["topic_name"] = temporal["topic_name"].str.slice(0, 32)
        st.plotly_chart(
            px.scatter(
                temporal,
                x="publication_year",
                y="predicted_trl",
                facet_col="topic_name",
                facet_col_wrap=3,
                color="source_code",
                size="topic_score",
                size_max=18,
                hover_data=["document_id", "title", "topic_score"],
                labels={
                    "publication_year": "Year",
                    "predicted_trl": "Estimated TRL",
                    "source_code": "Source",
                    "topic_name": "Topic",
                },
                title="Temporal positioning of documents by estimated TRL",
            ).update_yaxes(range=[0.5, 9.5], dtick=1),
            use_container_width=True,
            key="historical_topic_time_trl",
        )

    st.markdown("#### Document–keyword relational network")
    keyword_rows: list[tuple[str, str]] = []
    for row in documents[["document_id", "keywords"]].drop_duplicates().itertuples(index=False):
        if not isinstance(row.keywords, str):
            continue
        terms = {
            re.sub(r"\s+", " ", term.strip().lower())
            for term in re.split(r"[;,|]", row.keywords)
            if term.strip()
        }
        keyword_rows.extend((str(row.document_id), term) for term in terms)

    keyword_counts = Counter(term for _, term in keyword_rows)
    selected_keywords = {
        term for term, count in keyword_counts.most_common(30) if count >= 2
    }
    keyword_edges = [
        (document_id, term)
        for document_id, term in keyword_rows
        if term in selected_keywords
    ]
    if not keyword_edges:
        st.info("No reusable keyword links are available for the current filters.")
    else:
        node_labels = list(dict.fromkeys([doc for doc, _ in keyword_edges] + list(selected_keywords)))
        document_nodes = [node for node in node_labels if node not in selected_keywords]
        keyword_nodes = [node for node in node_labels if node in selected_keywords]
        positions = {}
        for index, node in enumerate(document_nodes):
            angle = 2 * pi * index / max(len(document_nodes), 1)
            positions[node] = (-0.55 + 0.18 * cos(angle), 0.9 * sin(angle))
        for index, node in enumerate(keyword_nodes):
            angle = 2 * pi * index / max(len(keyword_nodes), 1)
            positions[node] = (0.55 + 0.18 * cos(angle), 0.9 * sin(angle))

        edge_x: list[float] = []
        edge_y: list[float] = []
        for document_id, term in keyword_edges:
            edge_x.extend([positions[document_id][0], positions[term][0], None])
            edge_y.extend([positions[document_id][1], positions[term][1], None])
        figure = go.Figure(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=0.6, color="#C9D3DC"),
                hoverinfo="skip",
            )
        )
        for nodes, color, label in [
            (document_nodes, "#7E9BCB", "Documents"),
            (keyword_nodes, "#2A9D8F", "Keywords"),
        ]:
            figure.add_trace(
                go.Scatter(
                    x=[positions[node][0] for node in nodes],
                    y=[positions[node][1] for node in nodes],
                    mode="markers+text" if label == "Keywords" else "markers",
                    text=[node[:26] for node in nodes] if label == "Keywords" else None,
                    textposition="top center",
                    marker=dict(size=10 if label == "Documents" else 14, color=color),
                    name=label,
                    customdata=[[node] for node in nodes],
                    hovertemplate=f"{label[:-1]}: %{{customdata[0]}}<extra></extra>",
                )
            )
        figure.update_layout(
            title="Document–keyword relational network",
            height=700,
            showlegend=True,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.caption("Showing the 30 most frequent keywords with at least two document links.")
        st.plotly_chart(figure, use_container_width=True, key="historical_document_keyword")

    st.markdown("#### Intra-model thematic association network")
    st.caption(
        "The current reproducible version uses shared-document edges; this replaces "
        "the earlier Jaccard-edge exploratory plot."
    )
    render_relationships(topic_ids, key_prefix="historical_relationships")


def render_personalized_visualization(documents: pd.DataFrame) -> None:
    """Build an interactive chart from user-selected dimensions."""
    st.subheader("Personalized visualization")
    if documents.empty:
        st.info("No documents match the current filters.")
        return

    topic_options = documents[["topic_id", "topic_name"]].drop_duplicates().sort_values(
        "topic_name"
    )
    selected_ids = st.multiselect(
        "Topics to display",
        topic_options["topic_id"].tolist(),
        default=topic_options["topic_id"].tolist(),
        format_func=lambda value: topic_options.loc[
            topic_options.topic_id.eq(value), "topic_name"
        ].iloc[0],
        key="personalized_topics",
    )
    data = documents[documents["topic_id"].isin(selected_ids)].copy()
    if data.empty:
        st.info("Select at least one topic.")
        return

    dimension_labels = {
        "Topic": "topic_name",
        "Year": "publication_year",
        "Source": "source_code",
        "Estimated TRL": "predicted_trl",
    }
    chart_type = st.selectbox(
        "Plot type",
        ["Bar", "Line", "Scatter", "Heatmap"],
        key="personalized_plot_type",
    )
    x_label = st.selectbox(
        "X dimension",
        list(dimension_labels),
        index=1 if chart_type in {"Line", "Scatter"} else 0,
        key="personalized_x_dimension",
    )
    x_column = dimension_labels[x_label]

    if chart_type == "Scatter":
        numeric_dimensions = {
            "Year": "publication_year",
            "Estimated TRL": "predicted_trl",
            "Topic score": "topic_score",
        }
        y_label = st.selectbox(
            "Y dimension",
            list(numeric_dimensions),
            index=1,
            key="personalized_y_dimension",
        )
        y_column = numeric_dimensions[y_label]
        scatter_data = data.dropna(subset=[x_column, y_column]).copy()
        if scatter_data.empty:
            st.info("The selected dimensions have no complete observations.")
            return
        figure = px.scatter(
            scatter_data,
            x=x_column,
            y=y_column,
            color="topic_name",
            symbol="source_code",
            hover_data=["document_id", "title", "topic_score"],
            labels={
                x_column: x_label,
                y_column: y_label,
                "topic_name": "Topic",
                "source_code": "Source",
            },
            title=f"{y_label} by {x_label}",
        )
    elif chart_type == "Heatmap":
        y_label = st.selectbox(
            "Y dimension",
            [label for label in dimension_labels if label != x_label],
            key="personalized_heatmap_y_dimension",
        )
        y_column = dimension_labels[y_label]
        grouped = (
            data.groupby([x_column, y_column], dropna=False)["document_id"]
            .nunique()
            .reset_index(name="documents")
        )
        matrix = grouped.pivot(index=y_column, columns=x_column, values="documents").fillna(0)
        if matrix.empty:
            st.info("The selected dimensions have no complete observations.")
            return
        figure = px.imshow(
            matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": x_label, "y": y_label, "color": "Distinct documents"},
            title=f"Document count — {y_label} × {x_label}",
        )
    else:
        group_columns = [x_column]
        if x_column != "topic_name":
            group_columns.append("topic_name")
        aggregation = (
            data.groupby(group_columns, dropna=False)["document_id"]
            .nunique()
            .reset_index(name="documents")
            .dropna(subset=[x_column])
        )
        color_column = "topic_name" if x_column != "topic_name" else None
        if chart_type == "Bar":
            figure = px.bar(
                aggregation,
                x=x_column,
                y="documents",
                color=color_column,
                barmode="group",
                labels={
                    x_column: x_label,
                    "documents": "Distinct documents",
                    "topic_name": "Topic",
                },
                title=f"Distinct documents by {x_label}",
            )
        else:
            figure = px.line(
                aggregation,
                x=x_column,
                y="documents",
                color=color_column,
                markers=True,
                labels={
                    x_column: x_label,
                    "documents": "Distinct documents",
                    "topic_name": "Topic",
                },
                title=f"Distinct documents by {x_label}",
            )

    figure.update_layout(legend_title_text="Topic", height=650)
    st.plotly_chart(figure, use_container_width=True, key="personalized_visualization")


def render_landscape_views(documents: pd.DataFrame) -> None:
    st.subheader("Technology landscape and documentary sources")
    st.caption("Counts refer to distinct documents; topic assignments are multi-label, so one document may contribute to several topics.")
    distribution = topic_distribution(documents)
    if distribution.empty:
        st.info("No documents match the current filters.")
        return

    st.markdown("#### Topic prevalence")
    st.plotly_chart(
        px.bar(
            distribution.sort_values("documents"),
            x="documents",
            y="topic_label",
            orientation="h",
            labels={"documents": "Distinct documents", "topic_label": "Topic"},
        ),
        use_container_width=True,
        key="landscape_topic_prevalence",
    )

    source = (
        documents.groupby(["topic_name", "source_code"], as_index=False)["document_id"]
        .nunique()
        .rename(columns={"topic_name": "topic", "document_id": "documents"})
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Topic × source counts")
        heat = source.pivot(index="topic", columns="source_code", values="documents").fillna(0)
        st.plotly_chart(
            px.imshow(
                heat,
                text_auto=True,
                aspect="auto",
                labels={"x": "Source", "y": "Topic", "color": "Distinct documents"},
                color_continuous_scale="Blues",
            ),
            use_container_width=True,
            key="landscape_topic_source",
        )
    with right:
        st.markdown("#### Relative source composition")
        composition = source.copy()
        composition["share"] = composition["documents"] / composition.groupby("topic")["documents"].transform("sum")
        st.plotly_chart(
            px.bar(
                composition,
                x="topic",
                y="share",
                color="source_code",
                barmode="stack",
                labels={"topic": "Topic", "share": "Share of topic documents", "source_code": "Source"},
            ).update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-45),
            use_container_width=True,
            key="landscape_source_composition",
        )


def render_temporal_views(documents: pd.DataFrame) -> None:
    st.subheader("Temporal evolution and emergence")
    st.caption("Temporal counts are distinct documents per year and topic; lines are ordered chronologically.")
    temporal = (
        documents.dropna(subset=["publication_year"])
        .groupby(["publication_year", "topic_name"], as_index=False)["document_id"]
        .nunique()
        .rename(columns={"document_id": "documents", "topic_name": "topic"})
    )
    if temporal.empty:
        st.info("No publication years are available.")
        return

    st.markdown("#### Topic trajectories")
    st.plotly_chart(
        px.line(
            temporal,
            x="publication_year",
            y="documents",
            color="topic",
            markers=True,
            labels={"publication_year": "Year", "documents": "Distinct documents", "topic": "Topic"},
        ),
        use_container_width=True,
        key="temporal_topic_trajectories",
    )

    source_year = (
        documents.dropna(subset=["publication_year"])
        .groupby(["publication_year", "source_code"], as_index=False)["document_id"]
        .nunique()
        .rename(columns={"document_id": "documents"})
    )
    st.markdown("#### Documentary activity by source")
    st.plotly_chart(
        px.area(
            source_year,
            x="publication_year",
            y="documents",
            color="source_code",
            labels={"publication_year": "Year", "documents": "Distinct documents", "source_code": "Source"},
        ),
        use_container_width=True,
        key="temporal_source_activity",
    )

    recent_years = sorted(temporal["publication_year"].unique())[-3:]
    previous_years = sorted(temporal["publication_year"].unique())[:-3]
    if previous_years and recent_years:
        recent = temporal[temporal["publication_year"].isin(recent_years)].groupby("topic")["documents"].sum()
        previous = temporal[temporal["publication_year"].isin(previous_years)].groupby("topic")["documents"].sum()
        emergence = pd.DataFrame({"recent_documents": recent, "previous_documents": previous}).fillna(0)
        emergence["recent_share"] = emergence["recent_documents"] / max(len(recent_years), 1)
        emergence["previous_share"] = emergence["previous_documents"] / max(len(previous_years), 1)
        emergence["growth"] = emergence["recent_share"] - emergence["previous_share"]
        emergence = emergence.reset_index().sort_values("growth", ascending=False).head(15)
        st.markdown("#### Recent activity signal")
        st.caption("Activity signal = mean annual topic-document count in the last three years minus the mean annual count in all preceding years; exploratory, not a forecast.")
        st.plotly_chart(
            px.bar(
                emergence.sort_values("growth"),
                x="growth",
                y="topic",
                orientation="h",
                color="growth",
                color_continuous_scale="Cividis",
                labels={"growth": "Change in average annual volume", "topic": "Topic"},
            ),
            use_container_width=True,
            key="temporal_emergence_signal",
        )


def render_maturity_views(documents: pd.DataFrame) -> None:
    st.subheader("Maturity and TRL profiles")
    st.caption("TRL values are semantic evidence candidates, not verified maturity measurements.")
    maturity = documents.dropna(subset=["predicted_trl"]).copy()
    if maturity.empty:
        st.info("No TRL observations are available.")
        return
    maturity["topic"] = maturity["topic_name"]
    left, right = st.columns(2)
    with left:
        st.markdown("#### Estimated TRL distribution within each topic")
        st.caption("Each box summarizes the median, interquartile range and outliers of document-level estimated TRL values for a topic.")
        st.plotly_chart(
            px.box(
                maturity,
                x="topic",
                y="predicted_trl",
                color="topic",
                points="outliers",
                labels={"topic": "Topic", "predicted_trl": "Estimated TRL"},
            ).update_layout(showlegend=False, xaxis_tickangle=-45).update_traces(
                marker_color="#0072B2"
            ),
            use_container_width=True,
            key="maturity_topic_box",
        )
    with right:
        st.markdown("#### Topic volume × estimated TRL")
        st.caption("Each point is a topic: X = distinct documents, Y = mean estimated TRL across its documents, color = median estimated TRL.")
        profile = maturity.groupby("topic", as_index=False).agg(
            documents=("document_id", "nunique"),
            mean_trl=("predicted_trl", "mean"),
            median_trl=("predicted_trl", "median"),
        )
        st.plotly_chart(
            px.scatter(
                profile,
                x="documents",
                y="mean_trl",
                size="documents",
                color="median_trl",
                hover_data=["topic", "median_trl"],
                color_continuous_scale="Cividis",
                labels={"documents": "Distinct documents", "mean_trl": "Mean estimated TRL", "median_trl": "Median estimated TRL"},
            ),
            use_container_width=True,
            key="maturity_volume_mean_trl",
        )

    st.markdown("#### Topic × TRL profile")
    st.caption("Each row sums to 100%; cells show the percentage of documents in a topic assigned to each estimated TRL.")
    counts = maturity.groupby(["topic", "predicted_trl"], as_index=False)["document_id"].nunique()
    counts = counts.rename(columns={"document_id": "documents"})
    counts["share"] = counts["documents"] / counts.groupby("topic")["documents"].transform("sum")
    matrix = counts.pivot(index="topic", columns="predicted_trl", values="share").fillna(0)
    st.plotly_chart(
        px.imshow(
            matrix,
            text_auto=".0%",
            aspect="auto",
            zmin=0,
            zmax=1,
            color_continuous_scale="Cividis",
            labels={"x": "Estimated TRL", "y": "Topic", "color": "% of topic documents by TRL"},
        ),
        use_container_width=True,
        key="maturity_normalized_heatmap",
    )


def render_quality_views(documents: pd.DataFrame) -> None:
    st.subheader("Topic assignment quality and coverage")
    st.caption("These views describe multi-label topic coverage and assignment structure, not model accuracy.")
    if documents.empty:
        st.info("No documents match the current filters.")
        return

    per_document = documents.groupby("document_id").agg(
        topic_count=("topic_id", "nunique"),
        strongest_score=("topic_score", "max"),
    ).reset_index()
    st.markdown("#### Number of topics assigned per document")
    st.plotly_chart(
        px.histogram(
            per_document,
            x="topic_count",
            nbins=max(int(per_document["topic_count"].max()), 1),
            labels={"topic_count": "Number of assigned topics", "count": "Documents"},
        ),
        use_container_width=True,
        key="quality_topics_per_document",
    )

    matrix = documents.pivot_table(
        index="document_id",
        columns="topic_name",
        values="topic_score",
        aggfunc="max",
        fill_value=0,
    )
    if not matrix.empty:
        st.markdown("#### Document × topic assignment matrix")
        st.plotly_chart(
            px.imshow(
                matrix,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={"x": "Topic", "y": "Document", "color": "Topic score"},
            ),
            use_container_width=True,
            key="quality_document_topic_matrix",
        )


def render_text_views(documents: pd.DataFrame) -> None:
    st.subheader("Text content and characteristic terms")
    st.caption("Terms are ranked by TF-IDF within topic-level text aggregates; higher values indicate greater topic specificity.")
    text = documents.dropna(subset=["document_text", "topic_name"]).copy()
    text = text[text["document_text"].str.strip().ne("")]
    if text.empty:
        st.info("No text is available for the selected documents.")
        return

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.95)
    corpus = text.groupby("topic_name")["document_text"].agg(" ".join)
    matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()
    rows = []
    for row_index, topic in enumerate(corpus.index):
        scores = matrix[row_index].toarray().ravel()
        for term_index in scores.argsort()[-8:][::-1]:
            rows.append({"topic": topic, "term": terms[term_index], "tfidf": scores[term_index]})
    terms_df = pd.DataFrame(rows)
    st.markdown("#### Characteristic TF-IDF terms by topic")
    st.plotly_chart(
        px.bar(
            terms_df.sort_values(["topic", "tfidf"]),
            x="tfidf",
            y="term",
            color="topic",
            facet_col="topic",
            facet_col_wrap=3,
            orientation="h",
            labels={"tfidf": "Mean TF-IDF", "term": "Term", "topic": "Topic"},
        ),
        use_container_width=True,
        key="text_characteristic_terms",
    )

    st.markdown("#### Topic–keyword network")
    keyword_data = text[["topic_name", "keywords"]].dropna().copy()
    keyword_data["keyword"] = keyword_data["keywords"].str.split(r"[;,|]")
    keyword_data = keyword_data.explode("keyword")
    keyword_data["keyword"] = keyword_data["keyword"].astype(str).str.strip().str.lower()
    keyword_data = keyword_data[keyword_data["keyword"].ne("")]
    top_keywords = keyword_data["keyword"].value_counts().head(25).index
    keyword_data = keyword_data[keyword_data["keyword"].isin(top_keywords)]
    network_counts = keyword_data.groupby(["topic_name", "keyword"], as_index=False).size()
    st.dataframe(
        network_counts.sort_values("size", ascending=False).rename(
            columns={"topic_name": "Topic", "keyword": "Keyword", "size": "Documents"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_evidence(documents: pd.DataFrame) -> None:
    st.subheader("TRL evidence passages")
    if documents.empty:
        st.info("No documents match the current filters.")
        return
    document_ids = documents["document_id"].drop_duplicates().tolist()
    placeholders = ",".join("?" for _ in document_ids)
    evidence = read_query(
        f"""
        SELECT p.document_id, p.matched_trl, p.matched_label, p.similarity,
               p.retrieval_rank, p.passage
        FROM trl_evidence_passages p
        WHERE p.document_id IN ({placeholders})
          AND p.retrieval_rank IS NOT NULL
        ORDER BY p.similarity DESC
        LIMIT 500
        """,
        tuple(document_ids),
    )
    st.caption("The passages are semantic evidence candidates, not verified maturity proof.")
    st.dataframe(evidence, use_container_width=True, hide_index=True)
    st.download_button(
        "Download evidence passages",
        evidence.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_trl_evidence_passages.csv",
        mime="text/csv",
    )


st.title("ICED — Topic and maturity explorer")
st.caption("Read-only exploration of the fixed relational SQLite database. Sanity-check technology labels are not used.")

try:
    topics = load_topics()
    sources = load_sources()
    years = load_years()
except Exception as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Filters")
    selected_topic_ids = st.multiselect(
        "Topics",
        topics["topic_id"].tolist(),
        format_func=lambda value: topics.loc[topics.topic_id.eq(value), "topic_name"].iloc[0],
    )
    selected_sources = st.multiselect("Sources", sources, default=sources)
    selected_years = st.multiselect("Years", years, default=years)
    selected_trls = st.multiselect("Estimated TRL", list(range(1, 10)), default=list(range(1, 10)))
    topic_view = st.radio(
        "Topic representation",
        ["Combined", "Different colors", "Separate figure per topic"],
        index=1,
        help="Choose whether selected topics are aggregated, differentiated by color, or plotted separately.",
    )
    st.divider()
    st.write("The filters apply to topic-document observations. A document can appear more than once because topic assignments are multi-label.")

documents = filtered_documents(selected_topic_ids, selected_sources, selected_years, selected_trls)
metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Documents", documents["document_id"].nunique())
metric_b.metric("Topic observations", len(documents))
metric_c.metric("Sources", documents["source_code"].nunique())

tab_landscape, tab_temporal, tab_maturity, tab_relations, tab_documents, tab_quality, tab_text, tab_integrated, tab_personalized, tab_evidence = st.tabs(
    [
        "Landscape & sources",
        "Temporal evolution",
        "Maturity & TRL",
        "Relations & similarity",
        "Documents",
        "Topic quality",
        "Text & keywords",
        "Integrated view",
        "Personalized",
        "TRL evidence",
    ]
)
with tab_landscape:
    render_landscape_views(documents)
with tab_temporal:
    render_temporal_views(documents)
with tab_maturity:
    render_maturity_views(documents)
with tab_relations:
    render_relationships(selected_topic_ids or topics["topic_id"].tolist())
    render_graph_section(
        documents, selected_topic_ids or topics["topic_id"].tolist()
    )
with tab_documents:
    render_topic_explorer(documents)
with tab_quality:
    render_quality_views(documents)
with tab_text:
    render_text_views(documents)
with tab_integrated:
    render_integrated_view(documents)
with tab_personalized:
    render_personalized_visualization(documents)
with tab_evidence:
    render_evidence(documents)
