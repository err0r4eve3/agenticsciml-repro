#!/usr/bin/env node
import { createRequire } from "node:module";

const require = createRequire(new URL("../frontend/package.json", import.meta.url));
const { chromium } = require("playwright-core");

const options = parseArgs(process.argv.slice(2));
const browser = await chromium.launch({
  channel: options.channel,
  headless: options.headless,
});

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  page.setDefaultTimeout(options.timeoutMs);
  const websocketEvents = [];
  const codeServerConnectionErrors = [];
  page.on("console", (message) => {
    const text = message.text();
    if (/WebSocket|1006|Unexpected response code|workbench failed to connect|failed to connect to the server/i.test(text)) {
      codeServerConnectionErrors.push(text);
    }
  });
  page.on("websocket", (websocket) => {
    const event = { url: websocket.url(), closed: false };
    websocketEvents.push(event);
    websocket.on("close", () => {
      event.closed = true;
    });
  });
  await page.goto(options.baseUrl, { waitUntil: "domcontentloaded" });

  await expectVisible(page, "[data-testid='page-chat']", "Chat page did not render");
  await expectNotVisibleText(page, "[data-testid='page-chat']", "实验工作台", "Chat page leaked workbench copy");

  await sendChatMessage(page, "你是谁");
  await expectVisibleText(page, "[data-testid='chat-transcript']", "AgenticSciML 助手", "Ask identity reply missing");
  await expectNotVisibleText(page, "[data-testid='chat-transcript']", "actions", "Ask mode returned visible actions");

  await page.locator("[data-testid='assistant-mode-plan']").first().click();
  await sendChatMessage(page, "打开当前账号代码工作区");
  await expectVisibleText(page, "[data-testid='chat-transcript']", "open_code_server", "Plan mode did not expose proposed code action");
  await expectVisible(page, "[data-testid='page-chat']", "Plan mode should not navigate away from Chat page");

  await page.locator("[data-testid='assistant-mode-agent']").first().click();
  const codeServerWebsocketPromise = options.expectCodeServerWebsocket
    ? waitForCodeServerWebsocket(page, options.websocketTimeoutMs)
    : null;
  let verifiedCodeServerOpenWebsocketCount = null;
  await sendChatMessage(page, "打开当前账号代码工作区");
  await expectVisible(page, "[data-testid='page-ide']", "Agent code action did not navigate to IDE page");
  await expectVisible(page, "[data-testid='vscode-iframe']", "IDE page did not open the VS Code iframe");
  const codeServerWebsocketUrl = codeServerWebsocketPromise ? await codeServerWebsocketPromise : null;
  if (options.expectCodeServerWebsocket) {
    verifiedCodeServerOpenWebsocketCount = await assertCodeServerWorkbenchConnected(
      page,
      websocketEvents,
      codeServerConnectionErrors,
      options.websocketStabilizeMs,
    );
  }
  await expectNotVisibleText(page, "[data-testid='page-ide']", "实验工作台", "IDE page leaked workbench copy");
  await verifyIdeAgentPanelResizeAndCollapse(page);

  await page.locator("[data-testid='page-ide'] [data-testid='assistant-mode-ask']").click();
  await page.locator("[data-testid='ide-agent-input']").fill("你能做什么");
  await page.locator("[data-testid='ide-agent-send']").click();
  await expectVisibleText(page, "[data-testid='page-ide']", "Plan 模式", "IDE sidebar ChatUI did not answer in Ask mode");

  await page.locator("[data-testid='nav-library']").click();
  await expectVisible(page, "[data-testid='page-library']", "Library page did not render");
  await expectVisibleText(page, "[data-testid='page-library']", "实验工作台", "Library page should contain workbench copy");

  console.log(
    JSON.stringify(
      {
        ok: true,
        base_url: options.baseUrl,
        checks: [
          "chat_page_copy_boundary",
          "ask_no_actions",
          "plan_no_dispatch",
          "agent_open_code_server_dispatch",
          ...(options.expectCodeServerWebsocket ? ["code_server_websocket_stable"] : []),
          "ide_agent_panel_resize",
          "ide_agent_panel_collapse",
          "ide_sidebar_chatui_ask",
          "library_workbench_boundary",
        ],
        code_server_websocket_url: codeServerWebsocketUrl,
        observed_websocket_count: websocketEvents.length,
        verified_code_server_open_websocket_count: verifiedCodeServerOpenWebsocketCount,
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}

async function sendChatMessage(page, text) {
  await page.locator("[data-testid='chat-main-input']").fill(text);
  await page.locator("[data-testid='chat-main-send']").click();
}

async function expectVisible(page, selector, message) {
  await page.locator(selector).waitFor({ state: "visible" }).catch((error) => {
    throw new Error(`${message}: ${error.message}`);
  });
}

async function expectVisibleText(page, selector, text, message) {
  await page.locator(selector).getByText(text, { exact: false }).first().waitFor({ state: "visible" }).catch((error) => {
    throw new Error(`${message}: ${error.message}`);
  });
}

async function expectNotVisibleText(page, selector, text, message) {
  const count = await page.locator(selector).getByText(text, { exact: false }).count();
  if (count !== 0) {
    throw new Error(`${message}: found ${count} visible match(es) for ${JSON.stringify(text)}`);
  }
}

async function waitForCodeServerWebsocket(page, timeoutMs) {
  const websocket = await page.waitForEvent("websocket", {
    timeout: timeoutMs,
    predicate: (candidate) => {
      const url = candidate.url();
      return url.startsWith("ws://") || url.startsWith("wss://");
    },
  }).catch((error) => {
    throw new Error(`code-server WebSocket did not open through the page: ${error.message}`);
  });
  return websocket.url();
}

async function assertCodeServerWorkbenchConnected(page, websocketEvents, connectionErrors, stabilizeMs) {
  await page.waitForTimeout(stabilizeMs);
  const errorSample = connectionErrors.slice(0, 5);
  if (errorSample.length > 0) {
    throw new Error(`code-server WebSocket reported browser errors: ${JSON.stringify(errorSample)}`);
  }
  const openCodeServerSockets = websocketEvents.filter((event) => {
    return !event.closed && /\/stable-[^/?]+/.test(event.url);
  });
  if (openCodeServerSockets.length === 0) {
    throw new Error("code-server WebSocket opened but did not remain connected");
  }

  const iframe = await page.locator("[data-testid='vscode-iframe']").elementHandle();
  const frame = await iframe?.contentFrame();
  if (!frame) {
    throw new Error("VS Code iframe did not expose a browser frame");
  }
  const bodyText = await frame.locator("body").innerText({ timeout: 5000 });
  if (/An unexpected error occurred|workbench failed to connect|WebSocket close with status code 1006/i.test(bodyText)) {
    throw new Error(`code-server workbench displayed a connection failure: ${bodyText.slice(0, 300)}`);
  }
  return openCodeServerSockets.length;
}

async function verifyIdeAgentPanelResizeAndCollapse(page) {
  const panel = page.locator("[data-testid='ide-agent-panel']");
  const handle = page.locator("[data-testid='agent-resize-handle']");
  const panelBefore = await requiredBox(panel, "Agent panel did not expose a box before resize");
  const handleBox = await requiredBox(handle, "Agent panel resize handle did not expose a box");

  await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y + handleBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(handleBox.x - 96, handleBox.y + handleBox.height / 2, { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(160);

  const panelAfter = await requiredBox(panel, "Agent panel did not expose a box after resize");
  if (panelAfter.width < panelBefore.width + 48) {
    throw new Error(`Agent panel resize did not increase width enough: before=${panelBefore.width}, after=${panelAfter.width}`);
  }

  await page.getByLabel("收起 Agent 面板").click();
  await expectVisible(page, "[data-testid='ide-agent-collapsed']", "Agent panel did not collapse");
  const collapsedBox = await requiredBox(page.locator("[data-testid='ide-agent-collapsed']"), "Collapsed Agent rail did not expose a box");
  if (collapsedBox.width > 72) {
    throw new Error(`Collapsed Agent rail is too wide: ${collapsedBox.width}`);
  }

  await page.getByLabel("展开 Agent 面板").click();
  await expectVisible(page, "[data-testid='ide-agent-panel']", "Agent panel did not expand after collapse");
}

async function requiredBox(locator, message) {
  const box = await locator.boundingBox();
  if (!box) {
    throw new Error(message);
  }
  return box;
}

function parseArgs(args) {
  const parsed = {
    baseUrl: "http://127.0.0.1:5173",
    channel: "chrome",
    headless: true,
    timeoutMs: 30000,
    websocketTimeoutMs: 60000,
    websocketStabilizeMs: 15000,
    expectCodeServerWebsocket: false,
  };
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "--base-url") {
      parsed.baseUrl = requireValue(args, ++i, arg).replace(/\/$/, "");
    } else if (arg === "--channel") {
      parsed.channel = requireValue(args, ++i, arg);
    } else if (arg === "--timeout-ms") {
      parsed.timeoutMs = Number(requireValue(args, ++i, arg));
    } else if (arg === "--websocket-timeout-ms") {
      parsed.websocketTimeoutMs = Number(requireValue(args, ++i, arg));
    } else if (arg === "--websocket-stabilize-ms") {
      parsed.websocketStabilizeMs = Number(requireValue(args, ++i, arg));
    } else if (arg === "--expect-code-server-websocket") {
      parsed.expectCodeServerWebsocket = true;
    } else if (arg === "--headed") {
      parsed.headless = false;
    } else if (arg === "--headless") {
      parsed.headless = true;
    } else if (arg === "--help" || arg === "-h") {
      console.log("Usage: node scripts/web_ui_dispatch_e2e.mjs [--base-url URL] [--channel chrome] [--expect-code-server-websocket] [--headed]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function requireValue(args, index, option) {
  const value = args[index];
  if (!value) throw new Error(`${option} requires a value`);
  return value;
}
