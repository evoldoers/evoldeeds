import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  ScanCommand,
  PutCommand,
  GetCommand,
  DeleteCommand,
} from "@aws-sdk/lib-dynamodb";

const client = new DynamoDBClient({});

const dynamo = DynamoDBDocumentClient.from(client);

const familyTableName = "evoldeeds-families";
const historyTableName = "evoldeeds-histories";

export const handler = async (event, context) => {
  let body;
  let statusCode = 200;
  const headers = {
    "Content-Type": "application/json",
  };

  try {
    switch (event.routeKey) {
      case "GET /families/{id}":
        body = await dynamo.send(
          new GetCommand({
            TableName: familyTableName,
            Key: {
              id: event.pathParameters.id,
            },
          })
        );
        body = body.Item;
        break;
      case "GET /families":
        body = await dynamo.send(
          new ScanCommand({ TableName: familyTableName })
        );
        body = body.Items;
        break;
      case "GET /histories/{id}/{date}":
        body = await dynamo.send(
          new GetCommand({
            TableName: historyTableName,
            Key: {
              family_id: event.pathParameters.id,
              created: parseInt(event.pathParameters.date)
            },
          })
        );
        body = body.Item;
        break;
      case "POST /histories/{id}":
        let requestJSON = JSON.parse(event.body);
        const family_id = event.pathParameters.id;
        const created = Date.now();
        await dynamo.send(
          new PutCommand({
            TableName: historyTableName,
            Item: {
              family_id,
              created,
              history: requestJSON
            },
            ConditionExpression: 'attribute_not_exists(family_id)'
          })
        );
        body = {created}
        break;
      default:
        throw new Error(`Unsupported route: "${event.routeKey}"`);
    }
  } catch (err) {
    statusCode = 400;
    body = err.message;
  } finally {
    body = JSON.stringify(body);
  }

  return {
    statusCode,
    body,
    headers,
  };
};
