
# Shared list to store proverbs for auto-saving
generated_proverbs = []

async def fetch_proverb(semaphore, pbar):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "You are a wizened online denizen and a person who crafts pithy proverbs about modern life."},
                    {"role": "user", "content": "esCreate a proverb about life, especially as it occurs on the internet in social media, online forums, and other venues. Every proverb you generate must be a single, complete sentence up to 100 tokens."}
                ],
                temperature=1.1,
                max_tokens=60
            )
            proverb = response.choices[0].message.content.strip()
            
            # Add to list and handle auto-save
            generated_proverbs.append(proverb)
            pbar.update(1)
            
            if len(generated_proverbs) % SAVE_INTERVAL == 0:
                save_progress()
                
            return proverb
        except Exception as e:
            return None

def save_progress():
    # We use a set to ensure the saved file is always unique
    unique_data = list(set(generated_proverbs))
    with open(OUTPUT_FILE, "w") as f:
        json.dump(unique_data, f, indent=4)

async def main():
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    # Load existing proverbs if the script was interrupted
    global generated_proverbs
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            generated_proverbs = json.load(f)
            print(f"Resuming from {len(generated_proverbs)} existing proverbs...")

    remaining = TOTAL_PROVERBS - len(generated_proverbs)
    if remaining <= 0:
        print("Project already complete!")
        return

    with tqdm(total=remaining) as pbar:
        tasks = [fetch_proverb(semaphore, pbar) for _ in range(remaining)]
        await asyncio.gather(*tasks)
    
    # Final clean save
    save_progress()
    print(f"\nSuccessfully generated and saved {len(generated_proverbs)} proverbs.")


# Use this when running as script
if __name__ == "__main__":
    asyncio.run(main())