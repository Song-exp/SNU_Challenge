# -*- coding: utf-8 -*-
"""추론 스크립트: test.csv의 각 샘플(문장 + 프레임 4장)에 대해 프레임 순서를 예측하고
submission.csv를 생성한다. 베이스라인 노트북의 .py 포팅 버전.

사용 예:
    python src/inference.py
    python src/inference.py --data-dir data/snuaichallenge_data --output-dir outputs
"""
import argparse
import ast
import os

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


def get_prompt_message(row, image_dir):
    """4장의 프레임과 문장을 조합하여 모델에 보낼 프롬프트 메시지를 구성한다."""
    img_files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
    sentence = row["Sentence"]

    content = []
    for i, img_file in enumerate(img_files):
        img_path = os.path.join(image_dir, row["Id"], img_file)
        content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": f"\nImage {i + 1}\n"})

    user_text = (
        f'Thinking about the sentence: "{sentence}"\n'
        "Look at the 4 images above labeled Image 1 to Image 4. "
        "Determine the correct chronological order of these images to match the sentence. "
        "Provide the answer ONLY as a Python list of integers. "
        "Example: [1, 2, 3, 4]"
    )
    content.append({"type": "text", "text": user_text})

    return [{"role": "user", "content": content}]


def parse_model_output(output_text):
    """모델 출력(시간순 이미지 번호)을 제출 형식(각 이미지의 원래 위치)으로 역변환한다.

    예: 모델이 "[4, 2, 1, 3]" 출력 -> [3, 2, 4, 1] 반환. 파싱 실패 시 [1, 2, 3, 4].
    """
    try:
        start_idx = output_text.find("[")
        end_idx = output_text.rfind("]")

        if start_idx != -1 and end_idx != -1:
            result = ast.literal_eval(output_text[start_idx : end_idx + 1])

            if isinstance(result, list) and sorted(result) == [1, 2, 3, 4]:
                submission_answer = [0] * 4
                for index, image_num in enumerate(result):
                    submission_answer[image_num - 1] = index + 1
                return submission_answer
    except (ValueError, SyntaxError):
        pass

    return [1, 2, 3, 4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/snuaichallenge_data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--limit", type=int, default=0, help="앞에서부터 N개 샘플만 추론 (0 = 전체)")
    args = parser.parse_args()

    test_csv = os.path.join(args.data_dir, "test.csv")
    image_dir = os.path.join(args.data_dir, "test")
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    test_df = pd.read_csv(test_csv)
    if args.limit:
        test_df = test_df.head(args.limit)
    print(f"Test samples: {len(test_df)}")

    print(f"Loading model: {MODEL_NAME}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map=device
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        messages = get_prompt_message(row, image_dir)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        predictions.append({"Id": row["Id"], "Answer": str(parse_model_output(output_text))})

    submit_path = os.path.join(args.output_dir, "submission.csv")
    pd.DataFrame(predictions).to_csv(submit_path, index=False)
    print(f"Inference completed. Saved to {submit_path}")


if __name__ == "__main__":
    main()
