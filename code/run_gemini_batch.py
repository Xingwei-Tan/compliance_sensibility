from argparse import ArgumentParser
import yaml
import time
from google import genai
from google.genai import types


def get_model(file_path: str) -> str:
    if "gemini-3-flash-preview" in file_path:
        return "gemini-3-flash-preview"
    else:
        raise ValueError(f"Unrecognized model in file path: {file_path}")
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--file_path", type=str, default="batch_api/gemini-3-flash-preview_combined_router_test_0.jsonl")
    parser.add_argument("--batch_id", type=str, default="")
    parser.add_argument("--description", type=str, default="Eval reasoning type identification")
    args = parser.parse_args()


    with open(args.api_key, 'r') as file:
        api_key = yaml.safe_load(file)['google']
    client = genai.Client(api_key=api_key)


    model_name = get_model(args.file_path)

    if len(args.batch_id) == 0:
        batch_input_file = client.files.upload(
            file=args.file_path,
            config=types.UploadFileConfig(display_name=args.file_path.replace(".jsonl", "").split("/")[-1], mime_type="jsonl")
        )

        print("Batch input file uploaded:")
        print(batch_input_file)

        batch_return = client.batches.create(
            model=model_name,
            src=batch_input_file.name,
            config={
                'display_name': args.file_path.replace(".jsonl", "").split("/")[-1],
            },
        )
        batch_id = batch_return.name

        print("Batch API task created:")
        print(batch_return)
    else:
        batch_id = args.batch_id

    output_file = args.file_path.replace(".jsonl", "_response.jsonl")
    batch_status = client.batches.get(name=batch_id)
    print(batch_status)
    while batch_status.state.name not in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
        print(f"Batch status: {batch_status.state.name}")
        time.sleep(120)
        batch_status = client.batches.get(name=batch_id)

    if batch_status.state.name == "JOB_STATE_SUCCEEDED":
        result_file_name = batch_status.dest.file_name
        print(f"Results are in file: {result_file_name}")

        print("Downloading result file content...")
        file_response = client.files.download(file=result_file_name)
        # Process file_content (bytes) as needed
        response_text = file_response.decode('utf-8')


        with open(output_file, "w") as f:
            f.write(response_text)
        print(f"Output saved to {output_file}")
    else:
        print(f"Batch terminated with status: {batch_status.state.name}")
        exit(1)