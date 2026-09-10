import json
import re

transcript_path = "/Users/apple/.gemini/antigravity-ide/brain/99e2fe76-fbe2-4b7b-a61a-b7d6ceb26dcf/.system_generated/logs/transcript_full.jsonl"
csv_lines = []
found_csv = False

with open(transcript_path, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('type') == 'USER_INPUT' and 'Date,Month,Year' in data.get('content', ''):
                content = data['content']
                # Extract the CSV part
                csv_start = content.find('Date,Month,Year')
                if csv_start != -1:
                    csv_data = content[csv_start:]
                    # Stop at the end of the CSV (which is the end of the prompt in this case)
                    with open('/Users/apple/Downloads/AQI H/delhi_aqi.csv', 'w') as out:
                        out.write(csv_data.strip())
                    print("CSV extracted successfully.")
                    break
        except Exception as e:
            continue
