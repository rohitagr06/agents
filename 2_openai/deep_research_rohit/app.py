import gradio as gr
from research_manager import ResearchManager

async def async_run(query: str):

    output = ""
    async for chunk in ResearchManager().run(query):
        output += chunk
        yield output

def run(query: str):
    if not query.strip():
        return "Please enter a valid research query."
    return async_run(query)

with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")

    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

if __name__ == "__main__":
    ui.launch()