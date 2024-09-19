echo " Create wallets if not existent"
for name in Alice Bob Charlie Dave Eve Ferdie
do
echo "import bittensor as bt
w = bt.wallet(name='Alice', hotkey='$name')
if not w.coldkey_file.exists_on_device():
    w.create_coldkey_from_uri('//Alice', overwrite=True, use_password=False, suppress=True)
if not w.hotkey_file.exists_on_device():
    w.create_coldkey_from_uri('/$name', overwrite=True, use_password=False, suppress=False)
" > create_wallet.py
python3 create_wallet.py
rm create_wallet.py
done

# Close down all previous processes and restart them.
pm2 sendSignal SIGINT all
pm2 delete all

# Delete items from bucket
# python3 clean.py

# Start all the processes again.
pm2 start master.py --interpreter python3 --name MASTER -- --wallet.name Alice --wallet.hotkey Alice --subtensor.network test --device cuda:0 --use_wandb --restart
pm2 start validator.py --interpreter python3 --name VA -- --wallet.name Alice --wallet.hotkey Alice --subtensor.network test --device cuda:1 --use_wandb --restart
pm2 start miner.py --interpreter python3 --name M1 -- --wallet.name Alice --wallet.hotkey Charlie --subtensor.network test --device cuda:2 --use_wandb
pm2 start miner.py --interpreter python3 --name M2 -- --wallet.name Alice --wallet.hotkey Dave --subtensor.network test --device cuda:3 --use_wandb
pm2 start miner.py --interpreter python3 --name M3 -- --wallet.name Alice --wallet.hotkey Eve --subtensor.network test --device cuda:5 --use_wandb
pm2 start miner.py --interpreter python3 --name M4 -- --wallet.name Alice --wallet.hotkey Bob --subtensor.network test --device cuda:6 --use_wandb


# 
# python3 validator.py --wallet.name Alice --wallet.hotkey Alice --device cuda:0 --restart
# python3 miner.py --wallet.name Alice --wallet.hotkey Bob --device cuda:1 



