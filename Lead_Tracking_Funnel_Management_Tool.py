import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Use wide layout mode
st.set_page_config(page_title="Lead Tracking Funnel Tool", layout="wide")

# Load data from CSV files directly
def load_data_from_files():
    try:
        leads_data = pd.read_csv("leads_data.csv", parse_dates=["Last Contacted"])
    except FileNotFoundError:
        leads_data = pd.DataFrame(columns=["Name", "Email", "Phone", "Status", "Sales Rep", "Last Contacted"])
    try:
        interactions_data = pd.read_csv("interactions_data.csv", parse_dates=["Date"])
    except FileNotFoundError:
        interactions_data = pd.DataFrame(columns=["Lead", "Date", "Notes", "Status"])
    return leads_data, interactions_data

# Save data to CSV files
def save_data_to_files():
    st.session_state['leads_data'].to_csv("leads_data.csv", index=False)
    pd.DataFrame(st.session_state['interactions']).to_csv("interactions_data.csv", index=False)

# Initialize app with data from files
if 'leads_data' not in st.session_state:
    leads_data, interactions_data = load_data_from_files()
    st.session_state['leads_data'] = leads_data
    st.session_state['interactions'] = interactions_data.to_dict('records')

# Ensure essential columns exist
if 'Sales Rep' not in st.session_state['leads_data']:
    st.session_state['leads_data']['Sales Rep'] = "Unassigned"
if 'Last Contacted' not in st.session_state['leads_data']:
    st.session_state['leads_data']['Last Contacted'] = pd.to_datetime("2024-11-01")

# Page Title
st.title("Lead Tracking & Funnel Management Tool")
st.write("""
The **Lead Tracking & Funnel Management Tool** helps keep track of leads and team activity in one place. You can easily add or update leads, log interactions, and see how leads are moving through the sales process. Simple charts and summaries highlight what needs attention, making it easier to stay organized and on top of follow-ups.
""")

st.sidebar.title("Menu")
menu = st.sidebar.selectbox(
    "Select a Feature",
    [
        "Manage Leads", 
        "Track Interactions", 
        "Lead Funnel", 
        "Sales Metrics", 
        "Visual Insights"
    ],
    format_func=lambda x: {
        "Manage Leads": "📋 Manage Leads",
        "Track Interactions": "📝 Track Interactions",
        "Lead Funnel": "🔀 Lead Funnel",
        "Sales Metrics": "📊 Sales Metrics",
        "Visual Insights": "📈 Visual Insights"
    }.get(x, x)  # Default to x if no match
)


# Manage Leads Section (CRUD functionality)
if menu == "Manage Leads":
    st.header("📋 Manage Leads")
    st.write("""
The **Manage Leads** section allows you to efficiently organize and maintain your leads. 
You can add new leads with their contact details and status, update information as they progress 
through the sales pipeline, and remove leads that are no longer active. This section ensures that your lead data is always up-to-date, helping you stay organized and focus on the most promising opportunities.
""")


    # Display leads
    st.write("### Current Leads:")
    if not st.session_state['leads_data'].empty:
        st.dataframe(st.session_state['leads_data'], width=1200)
    else:
        st.info("No leads available. Add a new lead below.")

    # Create a new lead
    st.write("### Add New Lead:")
    with st.form("add_lead_form", clear_on_submit=True):
        name = st.text_input("Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        status = st.selectbox("Status", ["New", "Contacted", "Qualified", "Proposal Sent", "Closed - Won", "Closed - Lost"])
        submitted = st.form_submit_button("Add Lead")
        if submitted:
            new_lead = {"Name": name, "Email": email, "Phone": phone, "Status": status, "Sales Rep": "Unassigned", "Last Contacted": datetime.now()}
            st.session_state['leads_data'] = pd.concat([st.session_state['leads_data'], pd.DataFrame([new_lead])], ignore_index=True)
            save_data_to_files()
            st.success("Lead added successfully!")

    # Update or delete an existing lead
    st.write("### Update or Delete Lead:")
    if not st.session_state['leads_data'].empty:
        lead_to_edit = st.selectbox("Select a Lead to Edit/Delete", st.session_state['leads_data']["Name"])
        lead_index = st.session_state['leads_data'][st.session_state['leads_data']["Name"] == lead_to_edit].index[0]

        with st.form("edit_lead_form", clear_on_submit=True):
            name = st.text_input("Name", value=st.session_state['leads_data'].iloc[lead_index]["Name"])
            email = st.text_input("Email", value=st.session_state['leads_data'].iloc[lead_index]["Email"])
            phone = st.text_input("Phone", value=st.session_state['leads_data'].iloc[lead_index]["Phone"])
            status = st.selectbox(
                "Status",
                ["New", "Contacted", "Qualified", "Proposal Sent", "Closed - Won", "Closed - Lost"],
                index=["New", "Contacted", "Qualified", "Proposal Sent", "Closed - Won", "Closed - Lost"].index(
                    st.session_state['leads_data'].iloc[lead_index]["Status"]
                )
            )
            update = st.form_submit_button("Update Lead")
            delete = st.form_submit_button("Delete Lead")

            if update:
                st.session_state['leads_data'].iloc[lead_index] = [name, email, phone, status, st.session_state['leads_data'].iloc[lead_index]["Sales Rep"], datetime.now()]
                save_data_to_files()
                st.success("Lead updated successfully!")
            if delete:
                st.session_state['leads_data'] = st.session_state['leads_data'].drop(index=lead_index).reset_index(drop=True)
                save_data_to_files()
                st.success("Lead deleted successfully!")

# Other sections such as Track Interactions, Lead Funnel, Sales Metrics, and Visual Insights
# will follow the same logic of directly updating `st.session_state['leads_data']` where needed.
# Track Interactions Section
elif menu == "Track Interactions":
    st.header("📝 Track Interactions")
    st.write("Record interactions with leads and update their status.")

    if not st.session_state['leads_data'].empty:
        lead_name = st.selectbox("Select a Lead", st.session_state['leads_data']["Name"])
        interaction_date = st.date_input("Interaction Date", datetime.today())
        interaction_notes = st.text_area("Notes", "Enter details about the interaction...")
        lead_status = st.selectbox(
            "Update Lead Status",
            ["New", "Contacted", "Qualified", "Proposal Sent", "Closed - Won", "Closed - Lost"]
        )

        if st.button("Save Interaction"):
            interaction_data = {
                "Lead": lead_name,
                "Date": interaction_date,
                "Notes": interaction_notes,
                "Status": lead_status
            }
            st.session_state['interactions'].append(interaction_data)

            # Update lead status in leads data
            lead_index = st.session_state['leads_data'][st.session_state['leads_data']["Name"] == lead_name].index[0]
            st.session_state['leads_data'].iloc[lead_index]["Status"] = lead_status
            st.session_state['leads_data'].iloc[lead_index]["Last Contacted"] = interaction_date
            save_data_to_files()
            st.success("Interaction saved and lead status updated!")
    else:
        st.warning("No leads available. Add leads in the 'Manage Leads' section.")

# Lead Funnel Section
elif menu == "Lead Funnel":
    st.header("📊 Lead Funnel")
    st.write("Visualize your lead funnel based on the current lead statuses.")

    if not st.session_state['leads_data'].empty:
        status_counts = st.session_state['leads_data']["Status"].value_counts()

        # Funnel Data
        funnel_stages = ["New", "Contacted", "Qualified", "Proposal Sent", "Closed - Won", "Closed - Lost"]
        funnel_values = [status_counts.get(stage, 0) for stage in funnel_stages]

        # Create Funnel Chart
        fig = go.Figure(
            go.Funnel(
                y=funnel_stages,
                x=funnel_values,
                textinfo="value+percent initial",
                marker=dict(color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"])
            )
        )
        fig.update_layout(title="Dynamic Lead Funnel")
        st.plotly_chart(fig)
    else:
        st.info("No leads available. Add leads in the 'Manage Leads' section.")
        
# Sales Metrics Section
elif menu == "Sales Metrics":
    import pandas as pd
    import streamlit as st
    import numpy as np
    from datetime import datetime
    import pandasql as ps  # PandasSQL library
    from openai import OpenAI
    import re  # For query validation and extraction

    # Initialize OpenAI client with organization key
    # Initialize OpenAI client with organization key
    client = OpenAI(
        api_key="",
        organization=""
    )

    st.header("📈 Sales Metrics")
    st.write("""
    The **Sales Metrics** section gives sales managers a quick snapshot of key performance indicators like total leads, closed deals, and conversion rates. 
    It helps identify trends and bottlenecks in the sales process with intuitive metrics and visualizations. The integration with OpenAI's GPT-4o enables managers to ask natural language questions about the data, 
    unlocking actionable insights without needing technical expertise. This streamlines decision-making and empowers managers to focus on improving team performance.
    """)


    if not st.session_state.get('leads_data', pd.DataFrame()).empty:
        # Metrics Calculations
        total_leads = len(st.session_state['leads_data'])
        status_counts = st.session_state['leads_data']["Status"].value_counts()
        closed_won = int(status_counts.get("Closed - Won", 0))
        closed_lost = int(status_counts.get("Closed - Lost", 0))
        conversion_rate = round((closed_won / total_leads * 100), 2) if total_leads > 0 else 0.0

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Leads", total_leads)
        col2.metric("Deals Closed - Won", closed_won)
        col3.metric("Lost Deals", closed_lost)
        col4.metric("Conversion Rate", f"{conversion_rate:.2f}%")

        # Interactions Log
        interactions_df = pd.DataFrame(st.session_state['interactions'])

        if 'Date' in interactions_df.columns:
            interactions_df['Date'] = pd.to_datetime(interactions_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

        # Section to ask questions about interactions data
        st.subheader("🤖 Ask Questions About Interactions Data")
        st.write("Use AI to explore insights and trends in the interactions data.")

        # Dropdown with predefined questions
        question_options = [
            # Lead-Related Questions
            "📊 How many leads are in each status?",
            "📅 How many interactions were logged each day in the last 30 days?",
            "🔍 How many leads have not been contacted in the last 30 days, grouped by status?",
            "📌 How many new leads were added each week in the last 12 weeks?",
            "📈 How many leads progressed through each funnel stage (e.g., 'Contacted', 'Qualified', 'Proposal Sent')?",
            "🔁 Which leads had the most interactions in the last month?",
            "🕒 What is the average time it takes for leads to move between stages?",
            "🗓️ How many leads were contacted in the last 14 days, grouped by sales rep?",
            "📅 How many leads were added in each of the last 3 months?",
            "📈 How many leads moved to 'Proposal Sent' and 'Closed - Won' stages in the last 6 weeks?",

            # Additional Lead Movement Through Funnel Questions
            "📊 How many leads moved from 'New' to 'Contacted' in the last month?",
            "📅 How many leads progressed from 'Contacted' to 'Qualified' in the last 4 weeks?",
            "📈 How many leads moved from 'Qualified' to 'Proposal Sent' in the last 6 weeks?",
            "📊 How many leads moved from 'Proposal Sent' to 'Closed - Won' in the last 6 weeks?",
            "🔄 How many leads have moved from 'New' to 'Closed - Lost' in the last 3 months?",
            "📈 How many leads moved from 'New' to 'Qualified' in the last month?",
            "🕒 What is the average time for leads to progress from 'Contacted' to 'Proposal Sent'?",
            "📌 How many leads are currently in each funnel stage?",
            "🔁 How many leads that were marked as 'Qualified' are still in the funnel after 30 days?",
            "📊 How many leads have moved directly from 'New' to 'Closed - Won' in the last 3 months?",

            # Sales Rep Performance Questions
            "🌍 How many leads are assigned to each sales rep, grouped by status?",
            "📅 How many interactions were logged by each sales rep in the last 6 weeks?",
            "🔝 Which sales reps logged the most interactions in the last month?",
            "📈 How many leads did each sales rep move from 'New' to 'Contacted'?",
            "📝 How many leads were marked as 'Qualified' by each sales rep in the last month?",
            "🏆 How many 'Closed - Won' leads did each sales rep have in the last 6 weeks?",
            "📉 How many 'Closed - Lost' leads did each sales rep have in the last month?",
            "📬 How many overdue follow-ups are there for each sales rep?",
            "🕒 What is the average time between interactions for each sales rep in the last 6 weeks?",
            "📊 How many leads did each sales rep move to 'Proposal Sent' and 'Closed - Won' in the last 3 months?"
        ]


        selected_question = st.selectbox("Choose a question to analyze:", question_options)


        st.write("### Interactions Log")
        st.dataframe(interactions_df, width=1200)



        if st.button("Analyze Question"):
            def clean_query(query):
                """Clean and validate the SQL query."""
                query = re.sub(r"```[a-zA-Z]*\s*", "", query)
                query = query.strip()
                return query

            def validate_and_correct_query(query, valid_table_name="interactions_df"):
                """Ensure the query uses only valid table names."""
                table_names = re.findall(r"FROM\s+(\w+)", query, re.IGNORECASE)
                corrected_query = query
                for table in table_names:
                    if table != valid_table_name:
                        corrected_query = re.sub(
                            fr"\b{table}\b", valid_table_name, corrected_query, flags=re.IGNORECASE
                        )
                return corrected_query

            def generate_query(prompt):
                """Generate SQL query from OpenAI API."""
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a data analyst skilled in SQL and pandasql. Respond strictly with SQL query code."},
                        {"role": "user", "content": prompt}
                    ]
                )
                response_dict = response.model_dump()
                return response_dict["choices"][0]["message"]["content"].strip()

            def retry_query_with_error(query, error_message):
                """Retry query generation with error feedback."""
                retry_prompt = f"""
                The following SQL query failed with an error. Fix the query and ensure it references the table `interactions_df`.

                **Failed Query**:
                {query}

                **Error**:
                {error_message}

                Fix and return only the SQL query code.
                """
                return generate_query(retry_prompt)

            try:
                # Generate strict and specific prompt for OpenAI
                query_prompt = f"""
                You are a data analyst and an expert in SQL and pandasql. Generate only the SQL query code based on the following DataFrame schema and selected question. Ensure the query references the DataFrame `interactions_df`.

                **Schema**:
                {interactions_df.dtypes.to_string()}

                **First 5 Rows**:
                {interactions_df.head().to_string(index=False)}

                **Selected Question**:
                {selected_question}

                Only return the SQL query code. Do not include explanations, comments, or irrelevant text.
                """
                query_code = generate_query(query_prompt)

                # Clean, validate, and correct table names in the query
                query_code = clean_query(query_code)
                query_code = validate_and_correct_query(query_code)

                # Execute the generated query
                try:
                    result = ps.sqldf(query_code, {"interactions_df": interactions_df})
                    if result.empty:
                        st.warning("Query executed successfully but returned no results.")
                    else:
                        st.success("Query Processed Successfully!")
                        st.write(f"**Selected Question:** {selected_question}")
                        st.write(f"**Generated Query:**")
                        st.code(query_code, language="sql")
                        st.write("**Result:**")
                        st.write(result)

                        # Visualization function
                        import plotly.express as px  # For visualization

                        def generate_visualization(result_df):
                            """
                            Automatically generate a visualization based on the structure of the DataFrame.
                            """
                            if result_df.empty:
                                st.warning("The result DataFrame is empty. No visualization generated.")
                                return

                            st.subheader("📊 Visualization of Results")

                            # Determine type of visualization based on the DataFrame columns
                            columns = result_df.columns
                            if len(columns) >= 2:
                                x_col = columns[0]
                                y_col = columns[1]

                                # If there are multiple rows, use a bar chart
                                if len(result_df) > 1:
                                    fig = px.bar(result_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                                else:  # For single-row results, use a pie chart if appropriate
                                    fig = px.pie(result_df, names=x_col, values=y_col, title="Pie Chart Representation")
                            else:
                                st.info("Not enough data for a visualization.")
                                return

                            # Display the visualization
                            st.plotly_chart(fig)

                        # Replacing the visualization logic
                        if len(result) >= 3:
                            try:
                                # Generate visualization directly
                                generate_visualization(result)
                            except Exception as vis_gen_error:
                                st.error(f"Visualization generation failed: {vis_gen_error}")
                        else:
                            st.info("The result has fewer than 3 rows, so no visualization was generated.")

                except Exception as query_error:
                    # Retry logic
                    st.warning("Retrying query after error...")
                    retry_query_code = retry_query_with_error(query_code, str(query_error))
                    retry_query_code = clean_query(retry_query_code)
                    retry_query_code = validate_and_correct_query(retry_query_code)

                    try:
                        result = ps.sqldf(retry_query_code, {"interactions_df": interactions_df})
                        if result.empty:
                            st.warning("Retry executed successfully but returned no results.")
                        else:
                            st.success("Query Processed Successfully After Retry!")
                            st.write(f"**Selected Question:** {selected_question}")
                            st.write(f"**Generated Query (After Retry):**")
                            st.code(retry_query_code, language="sql")
                            st.write("**Result:**")
                            st.write(result)
                    except Exception as final_query_error:
                        st.error(f"Retry Failed: {final_query_error}")
                        st.markdown("### Original and Retried Query Code")
                        st.code(query_code, language="sql")
                        st.code(retry_query_code, language="sql")
            except Exception as error:
                st.error(f"An error occurred: {error}")


# Visual Insights Section
elif menu == "Visual Insights":
    st.header("📊 Visual Insights")

    # Distribution by Sales Rep
    st.subheader("Leads Distribution by Salesperson")
    sales_rep_counts = st.session_state['leads_data']["Sales Rep"].value_counts()
    pie_chart = px.pie(
        sales_rep_counts,
        names=sales_rep_counts.index,
        values=sales_rep_counts.values,
        title="Leads Assigned to Salespeople",
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(pie_chart, use_container_width=True)

    # Lead Aging Insights
    st.subheader("Lead Aging Insights")
    st.session_state['leads_data']["Days Since Last Contact"] = (
        datetime.now() - pd.to_datetime(st.session_state['leads_data']["Last Contacted"])
    ).dt.days
    follow_up_status = pd.cut(
        st.session_state['leads_data']["Days Since Last Contact"],
        bins=[-1, 7, 14, 30, float('inf')],
        labels=["Contacted Recently", "Follow-Up Soon", "Delayed Follow-Up", "Critical Overdue"]
    )
    st.session_state['leads_data']["Follow-Up Status"] = follow_up_status

    follow_up_summary = follow_up_status.value_counts()
    bar_chart = px.bar(
        x=follow_up_summary.index,
        y=follow_up_summary.values,
        title="Follow-Up Status Distribution",
        labels={"x": "Follow-Up Status", "y": "Number of Leads"}
    )
    st.plotly_chart(bar_chart, use_container_width=True)

    # Show actionable list of critical overdue leads
    st.subheader("Critical Overdue Leads (Needs Immediate Attention)")
    critical_leads = st.session_state['leads_data'][st.session_state['leads_data']["Follow-Up Status"] == "Critical Overdue"]
    if not critical_leads.empty:
        st.dataframe(
            critical_leads[["Name", "Sales Rep", "Last Contacted", "Days Since Last Contact"]],
            use_container_width=True
        )
    else:
        st.info("No critical overdue leads at the moment. Great job staying on top of follow-ups!")

    # Highlight actionable recommendations
    st.write("### Follow-Up Recommendations")
    st.write(
        "Prioritize leads categorized as 'Critical Overdue' and assign follow-up actions to sales reps. "
        "This will help ensure no potential deals fall through the cracks."
    )
