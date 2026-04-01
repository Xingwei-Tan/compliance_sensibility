from openai import OpenAI
from argparse import ArgumentParser
import yaml
import time


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--file_path", type=str, default="batch_api/gpt-51_combined_router_test_0.jsonl")
    parser.add_argument("--batch_id", type=str, default="")
    parser.add_argument("--description", type=str, default="Eval reasoning type identification")
    args = parser.parse_args()

    with open(args.api_key, 'r') as file:
        api_key = yaml.safe_load(file)['openai']
    client = OpenAI(api_key=api_key)

    if len(args.batch_id) == 0:
        batch_input_file = client.files.create(
            file=open(args.file_path, "rb"),
            purpose="batch"
        )

        print("Batch input file uploaded:")
        print(batch_input_file)

        batch_input_file_id = batch_input_file.id
        batch_return = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": args.description
            }
        )
        batch_id = batch_return.id

        print("Batch API task created:")
        print(batch_return)
        output_file = args.file_path.replace(".jsonl", "_response.jsonl")
    else:
        batch_id = args.batch_id
        output_file = f"batch_api/openai_{batch_id}_response.jsonl"

    batch_status = client.batches.retrieve(batch_id)
    print(batch_status)
    while batch_status.status not in ["completed", "failed", "expired", "cancelling", "cancelled"]:
        print(f"Batch status: {batch_status.status}")
        time.sleep(60)
        batch_status = client.batches.retrieve(batch_id)

    if batch_status.status == "completed":
        failed_count = batch_status.request_counts.failed
        if failed_count > 0:
            print(f"Batch completed with {failed_count} failed requests.")
            file_response = client.files.content(batch_status.error_file_id)
            print(file_response.read())
        else:
            print("Batch completed successfully with no failed requests.")
            file_response = client.files.content(batch_status.output_file_id)

            with open(output_file, "w") as f:
                f.write(file_response.text)
            print(f"Output saved to {output_file}")
    else:
        print(f"Batch terminated with status: {batch_status.status}")
        exit(1)