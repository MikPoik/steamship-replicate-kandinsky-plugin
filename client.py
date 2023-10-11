from steamship import Steamship

#from plugin_handle import PLUGIN_HANDLE
PLUGIN_HANDLE = "replicate-kandinsky-test"

text = """{input}"""

def main(prompt: str):
    with Steamship.temporary_workspace() as client:
        print(f'Running in workspace: {client.config.workspace_handle}')
        llm = client.use_plugin(PLUGIN_HANDLE,config={"replicate_api_key" : ""})
        msg = text.format(input=prompt)
        print(msg)
        task = llm.generate(text=msg, options={})
        task.wait()

        output_blocks = task.output.blocks
        for block in output_blocks:
            print(block.content_url)


if __name__ == "__main__":

    main((input("text:")))
