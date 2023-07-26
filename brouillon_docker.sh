#azure need subscription account, group and region 
docker context create aci myacicontext
#specify options in config file : --subscription-id, --resource-group, and --location

#print docker contexts 
docker context ls

#setting default context to distant 
docker context use myacicontext

#start docker app
docker compose up #use : docker-compose.yaml
#define conf : 
docker compose --file mycomposefile.yaml up
#specify a name for the Compose application
#--project-name
#list docker compose log 
docker compose ls
# ! These containers cannot be stopped, started, or removed independently since they are all part of the same ACI container group

#end docker app
docker compose down

#rm docker app 
docker compose down

#update the application by re-deploying it with the same project name
# new app : you will not be able to change CPU/memory reservation
docker compose --project-name PROJECT up

