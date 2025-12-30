# Baseline (15M)

Untuned run at 15M

val loss: 2.9858, val pplx: 19.8026, val bpb: 1.1339

hellaswag bpb remained around 3.75-3.8 the whole time, no zero-shot capabilities at this stage

<img width="674" height="374" alt="image" src="https://github.com/user-attachments/assets/465f4b0b-65ff-44e6-808c-3a5ef571c862" />

New MuP tuned baseline, 

val loss: 2.4943, val pplx: 12.1128, val bpb: 0.9472

reached untuned baseline 2.7x faster

<img width="658" height="374" alt="image" src="https://github.com/user-attachments/assets/db3b4113-e8d7-427d-ad04-ebd17d48659f" />

## MuP sweeps

Initial LR sweep, LR, dropout, Weight Decay

<img width="2238" height="1055" alt="image" src="https://github.com/user-attachments/assets/c99df370-4c6c-4bb0-a239-2d44442bf84c" />

Tested multiple LRs, found a optimal value for the 0.74M model, the winning LR was 3e-3, it might be due to scale, will test at 15M and 100M

<img width="7200" height="3000" alt="lr_s_split_saturated" src="https://github.com/user-attachments/assets/5e444a5e-c645-42b7-aa98-c6d49d42fb38" />

