# Close down all previous processes and restart them.
pm2 sendSignal SIGINT all
pm2 delete all

# Start all the processes again.
pm2 start validator.py --interpreter python3 --name Validator -- --wallet.name Alice --wallet.hotkey Alice --subtensor.network test --device cuda:1 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner1 -- --wallet.name Alice --wallet.hotkey Bob --subtensor.network test --device cuda:2 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner2 -- --wallet.name Alice --wallet.hotkey Charlie --subtensor.network test --device cuda:3 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner3 -- --wallet.name Alice --wallet.hotkey Dave --subtensor.network test --device cuda:4 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner4 -- --wallet.name Alice --wallet.hotkey Eve --subtensor.network test --device cuda:5 --use_wandb
pm2 start miner.py --interpreter python3 --name Baseline -- --wallet.name Alice --wallet.hotkey Alice --subtensor.network test --device cuda:6 --use_wandb

