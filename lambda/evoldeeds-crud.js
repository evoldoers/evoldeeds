import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  ScanCommand,
  PutCommand,
  GetCommand,
  DeleteCommand,
} from "@aws-sdk/lib-dynamodb";

import fs from 'fs';
import { expandCigarTree, countGapSizes, doLeavesMatchSequences } from './cigartree.js';
import { subLogLike, transLogLike, sum } from './likelihood.js';

const modelFilename = 'model.json';

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
        body = body.Items.map ((item) => item.id);
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
        const family_id = event.pathParameters.id;
        const family = await dynamo.send(
            new GetCommand({
              TableName: familyTableName,
              Key: {
                id: family_id,
              },
            })
          );
        const seqById = family.Item.seqById;
        
        const history = JSON.parse(event.body);
        const expandedHistory = expandCigarTree (history, seqById);
        if (!doLeavesMatchSequences (expandedHistory, seqById))
            throw new Error ("History does not match sequences");

        const { alignment, expandedCigar, distanceToParent, leavesByColumn, internalsByColumn, branchesByColumn } = expandedHistory;
        const { transCounts } = countGapSizes (expandedCigar);

        const modelJson = JSON.parse (fs.readFileSync(modelFilename).toString());
        const { alphabet, hmm, mixture } = modelJson;

        const { evecs_l, evals, evecs_r, root } = mixture[0];
        const subll = subLogLike (alignment, distanceToParent, leavesByColumn, internalsByColumn, branchesByColumn, alphabet, root, { evecs_l, evals, evecs_r });
        const subll_total = sum (subll);

        const transll = transLogLike (transCounts, distanceToParent, hmm);
        const transll_total = sum (transll);

        const score = subll_total + transll_total;

        const created = Date.now();
        await dynamo.send(
          new PutCommand({
            TableName: historyTableName,
            Item: {
              family_id,
              created,
              history,
              score
            },
            ConditionExpression: 'attribute_not_exists(family_id)'
          })
        );

        let newBestScore = false;
        if (!family.Item.score || score > family.Item.score)
            try {
                await dynamo.send(
                    new PutCommand({
                    TableName: familyTableName,
                    Item: {
                        ...family.Item,
                        history: { created },
                        score
                    },
                    ConditionExpression: 'attribute_not_exists(#score) OR #score < :newScore',
                    ExpressionAttributeNames: { '#score': 'score' },
                    ExpressionAttributeValues: { ':newScore': score },
                    })
                );
                newBestScore = true;
            } catch (e) { }
      
        body = {created, score}
        if (newBestScore)
            body.newBest = true;
        break;
      default:
        throw new Error(`Unsupported route: "${event.routeKey}"`);
    }
  } catch (err) {
    statusCode = 400;
    body = err.message + "\n" + err.stack;
  } finally {
    body = JSON.stringify(body);
  }

  return {
    statusCode,
    body,
    headers,
  };
};
