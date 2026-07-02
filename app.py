"""
app.py
------
Smart Restaurant Menu Assistant
An AI-powered Streamlit application that helps customers discover the
perfect dish based on their personal preferences using a KNN-based
recommendation engine (see model.py).

Run with:  streamlit run app.py
"""

import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from model import MenuRecommender

# ----------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLING
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Restaurant Menu Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a modern restaurant look & feel (dark maroon + gold theme)
CUSTOM_CSS = """
<style>
    .main {
        background-color: #FFFBF5;
    }
    h1, h2, h3 {
        font-family: 'Georgia', serif;
        color: #6E1423;
    }
    .menu-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(110, 20, 35, 0.10);
        border-left: 6px solid #D4A017;
        transition: transform 0.15s ease-in-out;
    }
    .menu-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(110, 20, 35, 0.18);
    }
    .price-tag {
        background: #6E1423;
        color: #FFFBF5;
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 6px;
    }
    .badge-veg { background: #DCF5E0; color: #1E7B34; }
    .badge-nonveg { background: #FBDCDC; color: #B3261E; }
    .badge-spice { background: #FFE6C7; color: #A85D00; }
    .badge-cuisine { background: #E7E4FF; color: #4B3FA8; }
    .match-score {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6E1423;
    }
    section[data-testid="stSidebar"] {
        background-color: #6E1423;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFBF5 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# LOAD DATA / MODEL (cached so it is not reloaded on every interaction)
# ----------------------------------------------------------------------
@st.cache_resource
def load_recommender():
    return MenuRecommender("menu_data.csv")


recommender = load_recommender()

# Session state to keep track of every recommendation ever shown, so the
# Analytics Dashboard can display "Most Recommended Dishes" for this session.
if "recommend_history" not in st.session_state:
    st.session_state.recommend_history = []

if "last_recommendations" not in st.session_state:
    st.session_state.last_recommendations = pd.DataFrame()

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.markdown("## 🍽️ Smart Menu Assistant")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📋 Interactive Menu",
        "🎯 Get Recommendations",
        "📊 Analytics Dashboard",
        "ℹ️ About",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • Scikit-learn • Plotly")


# ----------------------------------------------------------------------
# HELPER: render one menu item as a styled card
# ----------------------------------------------------------------------
def render_menu_card(row, show_match=False):
    veg_badge = (
        '<span class="badge badge-veg">🌱 Veg</span>'
        if row["Type"] == "Veg"
        else '<span class="badge badge-nonveg">🍗 Non-Veg</span>'
    )
    spice_badge = f'<span class="badge badge-spice">🌶️ {row["Spice_Level"]}</span>'
    cuisine_badge = f'<span class="badge badge-cuisine">🌍 {row["Cuisine"]}</span>'

    match_html = ""
    if show_match and "Match_Score" in row:
        match_html = (
            f'<div class="match-score">🎯 Match: {row["Match_Score"]}%</div>'
        )
        st_progress_value = row["Match_Score"] / 100
    else:
        st_progress_value = None

    card_html = f"""
    <div class="menu-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h4 style="margin:0;">{row['Item_Name']}</h4>
            <span class="price-tag">₹{row['Price']}</span>
        </div>
        <p style="color:#555; margin:8px 0 4px 0;">{row['Description']}</p>
        <div>{veg_badge}{spice_badge}{cuisine_badge}</div>
        <div style="margin-top:8px;">⭐ {row['Rating']} / 5.0</div>
        {match_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    if st_progress_value is not None:
        st.progress(st_progress_value)


# ----------------------------------------------------------------------
# PAGE: HOME
# ----------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🍽️ Smart Restaurant Menu Assistant")
    st.subheader("AI-powered dish recommendations, tailored to your taste")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Dishes", len(recommender.df))
    col2.metric("Categories", recommender.df["Category"].nunique())
    col3.metric("Cuisines", recommender.df["Cuisine"].nunique())
    col4.metric("Avg. Rating", f"{recommender.df['Rating'].mean():.1f} ⭐")

    st.markdown("---")
    st.markdown(
        """
        ### Welcome! 👋
        This assistant uses a **K-Nearest Neighbors (KNN)** machine learning
        model to recommend dishes that best match your personal taste —
        cuisine, spice tolerance, dietary preference and budget.

        **Use the sidebar to:**
        - 📋 Browse the full interactive menu with search & filters
        - 🎯 Fill in your preferences and get personalized AI recommendations
        - 📊 Explore menu analytics and popularity trends
        """
    )
    st.info("Tip: Start with **🎯 Get Recommendations** to see the AI in action!")


# ----------------------------------------------------------------------
# PAGE: INTERACTIVE MENU
# ----------------------------------------------------------------------
elif page == "📋 Interactive Menu":
    st.title("📋 Interactive Restaurant Menu")

    search_col, filter_col = st.columns([2, 1])
    with search_col:
        search_term = st.text_input("🔍 Search menu items", placeholder="e.g. paneer, pizza, chai...")
    with filter_col:
        category_filter = st.selectbox(
            "Filter by category", ["All"] + recommender.get_categories()
        )

    filtered_menu = recommender.filter_menu(search_term, category_filter)
    st.caption(f"Showing {len(filtered_menu)} of {len(recommender.df)} dishes")

    # Group by category for a nicer browsing experience (unless a search filters things down)
    categories_to_show = (
        [category_filter] if category_filter != "All" else recommender.get_categories()
    )

    for cat in categories_to_show:
        cat_items = filtered_menu[filtered_menu["Category"] == cat]
        if cat_items.empty:
            continue
        st.markdown(f"### {cat}")
        cols = st.columns(2)
        for i, (_, row) in enumerate(cat_items.iterrows()):
            with cols[i % 2]:
                render_menu_card(row)


# ----------------------------------------------------------------------
# PAGE: GET RECOMMENDATIONS
# ----------------------------------------------------------------------
elif page == "🎯 Get Recommendations":
    st.title("🎯 Tell Us Your Preferences")
    st.caption("Our AI (KNN model) will find the dishes that match you best.")

    with st.form("preference_form"):
        c1, c2 = st.columns(2)
        with c1:
            pref_cuisine = st.selectbox("Preferred Cuisine", recommender.get_cuisines())
            pref_spice = st.select_slider("Spice Level", options=["Low", "Medium", "High"], value="Medium")
            pref_category = st.selectbox("Category (optional)", ["All"] + recommender.get_categories())
        with c2:
            pref_type = st.radio("Dietary Preference", ["Veg", "Non-Veg", "Both"], horizontal=True)
            budget_range = st.slider(
                "Budget Range (₹)",
                min_value=int(recommender.df["Price"].min()),
                max_value=int(recommender.df["Price"].max()),
                value=(100, 300),
            )
            num_results = st.slider("Number of recommendations", 3, 10, 5)

        submitted = st.form_submit_button("✨ Get My Recommendations")

    if submitted:
        results = recommender.recommend(
            cuisine=pref_cuisine,
            spice_level=pref_spice,
            veg_type=pref_type,
            budget_min=budget_range[0],
            budget_max=budget_range[1],
            category=pref_category,
            top_n=num_results,
        )
        st.session_state.last_recommendations = results

        if results.empty:
            st.warning("No dishes match those filters. Try widening your budget range or category.")
        else:
            # Log this run into session history (for the analytics dashboard)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for _, r in results.iterrows():
                st.session_state.recommend_history.append(
                    {"Item_Name": r["Item_Name"], "Category": r["Category"], "Time": timestamp}
                )

            st.success(f"Found {len(results)} great matches for you!")
            cols = st.columns(2)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    render_menu_card(row, show_match=True)

    # Download report button (works on the most recent recommendation set)
    if not st.session_state.last_recommendations.empty:
        st.markdown("---")
        report_lines = ["SMART RESTAURANT MENU ASSISTANT — RECOMMENDATION REPORT"]
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 55)
        for _, row in st.session_state.last_recommendations.iterrows():
            report_lines.append(
                f"\n{row['Item_Name']}  (Match: {row['Match_Score']}%)"
                f"\n  Category : {row['Category']}"
                f"\n  Cuisine  : {row['Cuisine']}"
                f"\n  Price    : Rs.{row['Price']}"
                f"\n  Spice    : {row['Spice_Level']}"
                f"\n  Type     : {row['Type']}"
                f"\n  Rating   : {row['Rating']} / 5.0"
            )
        report_text = "\n".join(report_lines)

        st.download_button(
            label="📥 Download Recommendation Report (.txt)",
            data=report_text,
            file_name="menu_recommendations.txt",
            mime="text/plain",
        )


# ----------------------------------------------------------------------
# PAGE: ANALYTICS DASHBOARD
# ----------------------------------------------------------------------
elif page == "📊 Analytics Dashboard":
    st.title("📊 Menu Analytics Dashboard")

    tab1, tab2, tab3 = st.tabs(["Category & Cuisine", "Budget Analysis", "Most Recommended"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            cat_counts = recommender.df["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_cat = px.pie(
                cat_counts, names="Category", values="Count",
                title="Menu Category Distribution",
                color_discrete_sequence=px.colors.sequential.RdBu,
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        with c2:
            cui_counts = recommender.df["Cuisine"].value_counts().reset_index()
            cui_counts.columns = ["Cuisine", "Count"]
            fig_cui = px.bar(
                cui_counts, x="Cuisine", y="Count", color="Cuisine",
                title="Dishes per Cuisine",
            )
            st.plotly_chart(fig_cui, use_container_width=True)

    with tab2:
        fig_price = px.histogram(
            recommender.df, x="Price", nbins=15, color="Category",
            title="Price Distribution Across Menu",
            labels={"Price": "Price (₹)"},
        )
        st.plotly_chart(fig_price, use_container_width=True)

        fig_box = px.box(
            recommender.df, x="Category", y="Price", color="Category",
            title="Budget Range by Category",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        if st.session_state.recommend_history:
            hist_df = pd.DataFrame(st.session_state.recommend_history)
            top_recommended = (
                hist_df["Item_Name"].value_counts().reset_index().head(10)
            )
            top_recommended.columns = ["Item_Name", "Times_Recommended"]
            fig_top = px.bar(
                top_recommended, x="Item_Name", y="Times_Recommended",
                title="Most Recommended Dishes (This Session)",
                color="Times_Recommended", color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info(
                "No recommendations generated yet in this session. "
                "Visit '🎯 Get Recommendations' to build up analytics data!"
            )

    st.markdown("---")
    st.metric("Highest Rated Dish", recommender.df.loc[recommender.df["Rating"].idxmax(), "Item_Name"])


# ----------------------------------------------------------------------
# PAGE: ABOUT
# ----------------------------------------------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        **Smart Restaurant Menu Assistant** is an intermediate-level,
        AI-powered Streamlit project built to demonstrate:

        - Data handling with **Pandas / NumPy**
        - A **K-Nearest Neighbors** recommendation model with **Scikit-learn**
        - Interactive dashboards with **Plotly**
        - Clean, responsive UI/UX design in **Streamlit**

        ### How the AI Works
        1. Each menu item is encoded into a numeric feature vector
           (cuisine, spice level, veg/non-veg type, normalized price).
        2. Your preferences are encoded the same way.
        3. A KNN model finds the dishes whose vectors are closest to yours.
        4. The distance is converted into an easy-to-read **match percentage**.

        ---
        *This project was built as part of an internship submission.*
        """
    )
