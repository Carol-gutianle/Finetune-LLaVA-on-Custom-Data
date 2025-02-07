# Finetune-LLaVA-on-Custom-Data
This is a simple demo for how to fine-tune llava for a image classifier.

You can find the demo in `demo.json`. Here is the data format:
```
{
    "id": "db014d85-ba7f-4ec2-a83c-d00e5675df32",
    "image": "imgs/0540.jpg",
    "conversations": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please give a lable to the image."
                },
                {
                    "type": "image"
                }
            ]
        },
        {
            "role": "evaluator",
            "content": [
                {
                    "type": "text",
                    "text": "2"
                }
            ]
        }
    ]
}
```
