

let bodyPose;
let connections;

// How often to update the matching image (in milliseconds)
// Lower = more responsive but more CPU usage
// Higher = less CPU but slower updates
// Recommended: 250 (4x/sec) to 1000 (1x/sec)
let MATCH_UPDATE_INTERVAL_MS = 250;
let lastMatchTime = 0;

// Reference pose data loaded from JSON (array of objects with l2_vector and metadata)
let referencePoseData = null;

// Cache for loaded images (keyed by filename)
let imageCache = {};

// Webcam and pose matching variables
let video;
let webcamPoses = [];
let primaryPose = null;       // The pose we match against (largest person in frame)
let bestMatchImg = null;
let bestMatchData = null;      // Match we are switching to (image may still be loading)
let displayedMatchData = null; // Match whose image is actually on screen — the
                               // label, QR code, and announcements follow this so
                               // they never describe an image that hasn't loaded yet
let referenceList = null;      // Pre-processed reference poses (built on first match)

// QR code variables
let qrCode = null;
let currentQRObjectId = null;  // Track what QR is currently showing

// Where QR codes send a visitor's phone. Leave "" to link to wherever this
// page is served from (right for any public deployment). Set it to a public
// URL ending in "/" only when this copy is served from an address phones
// can't reach, such as an offline kiosk at http://localhost — the public
// site must then serve the same dataset.
const QR_BASE_URL = "";

// Kiosk mode, for an unattended exhibit running in a full-screen browser
// with no tabs, address bar or back button. Set to true and nothing on the
// page leads away from it: clicking the artwork does nothing, the About link
// is hidden, and the logo and footer links are inert. (Any of those would
// open a page or tab the visitor has no way to close, and an offline kiosk
// can't load the outside ones at all.) Visitors still get to the artwork's
// info page on their own phone through the QR code.
const KIOSK_MODE = false;

// Accessibility: debounce timer for screen reader announcements
let announceTimeout = null;
const ANNOUNCE_DELAY_MS = 1500;  // Wait 1.5 seconds of stable match before announcing

// Camera status
let cameraReady = false;
let cameraError = false;
let cameraErrorTitle = "";   // shown in place of the webcam feed
let cameraErrorDetail = "";

// Webcam crop parameters (set during draw, used to map skeleton keypoints)
let cropSx = 0, cropSy = 0, cropSw = 640, cropSh = 480;
let cropAreaW = 400, cropAreaH = 593;

// Explore mode: penalize recently shown images to encourage variety
// Without this, the same ~15 images dominate because ~71% of the dataset
// shares a nearly identical standing pose (cosine similarity > 0.95)
let exploreMode = true;
let recentMatches = [];
// How many recently shown images to remember and exclude from matching.
// Once an image falls off this list, it can be shown again.
// Sized close to the dataset (~1200 images) so a long session cycles
// through most of the collection before repeating.
// Recent images are excluded outright rather than score-penalized: for
// distinctive poses the dataset has only a few close matches and similarity
// falls off a cliff (0.1-0.45) right after them, so no fixed penalty can
// stop those few from winning over and over.
const EXPLORE_HISTORY_SIZE = 400;

// In explore mode, instead of always showing the single highest-scoring image,
// pick randomly among all near-tie candidates. For common poses, hundreds of
// images score within 0.02 of the top match — a difference no user can
// perceive — so sampling from that band spreads matches across the whole
// collection instead of the same few argmax winners.
const MATCH_SAMPLE_EPSILON = 0.015;  // candidates within this of the best score
const MATCH_SAMPLE_TOP_K = 40;       // cap on the candidate pool per pick

// Explore mode never shows an image scoring more than this far below the
// best match for the visitor's current pose. For common poses hundreds of
// images clear that floor, so it never comes into play; for a distinctive
// pose only a few dozen do, and without a floor explore would work through
// those and then slide into poor matches. Once every image above the floor
// has been shown, explore cycles back through them, least recently shown
// first, instead of going lower.
const EXPLORE_SIMILARITY_FLOOR = 0.05;

// Movement-aware timing for explore mode:
// Instead of penalizing images immediately, wait a minimum display time
// so the user can see each match before it rotates away.
let matchStartTime = 0;        // When the current match first appeared
let matchPoseVector = null;    // The visitor's pose at the moment the current match was chosen
let matchPoseMask = null;      // ...and which of its keypoints were confident
// Minimum time (ms) to show each image before it can be replaced.
// While the visitor holds the pose that produced the current image, it stays
// longer; once their pose has changed, it can be replaced quickly.
const EXPLORE_MIN_DISPLAY_STILL_MS = 5000;   // 5 seconds while holding the same pose
const EXPLORE_MIN_DISPLAY_MOVING_MS = 1000;  // 1 second once the pose has changed
// Cosine similarity between the visitor's current pose and their pose when
// the on-screen image was chosen. Below this, the pose counts as changed.
// (Comparing against the previous frame instead would treat someone who
// strikes a new pose and freezes as "still", and make them wait the full
// still time to see the match for it.)
// Only keypoints confident in both poses are compared. The model's guesses
// for joints it can't see jump around even on a motionless person — with
// legs out of frame that noise alone used up most of the margin below — and
// they dilute real movement of the joints that are visible.
const MOVEMENT_THRESHOLD = 0.98;

// Keypoints below this confidence are ignored for matching and drawing —
// MoveNet reports essentially random positions for occluded joints.
const KEYPOINT_CONFIDENCE_MIN = 0.1;
// Require at least this many confident keypoints before matching at all.
// Fewer means the pose vector is mostly noise (person half out of frame).
const MIN_CONFIDENT_KEYPOINTS = 8;

// Attract mode: what the exhibit shows when nobody is using it. It is on
// from page load, and comes back once nobody has been matchable for
// ATTRACT_AFTER_MS — long enough for a departing visitor to scan the QR code
// of their last match. While it is on, the artwork half shows a slideshow of
// random artworks (with their usual label and QR code) and the webcam half
// shows ATTRACT_PROMPT. It ends the moment a person with enough confident
// keypoints appears. Entering it also clears the explore history, so each
// new visitor starts fresh with the collection's best matches available.
const ATTRACT_AFTER_MS = 30000;
const ATTRACT_SLIDE_MS = 6000;      // how long each slideshow artwork stays up
const ATTRACT_PROMPT = "Step in and strike a pose";
let attractMode = true;
let attractNextSlideTime = 0;       // millis() when the slideshow next advances
let lastMatchableTime = null;       // millis() when a matchable pose was last seen

// When someone is in view but too few of their keypoints are confident to
// match (usually standing too close), say so instead of silently doing
// nothing. Needs at least TOO_CLOSE_MIN_KEYPOINTS so that a stray detection
// on an empty scene doesn't trigger it, for TOO_CLOSE_HINT_AFTER_MS.
const TOO_CLOSE_HINT = "Step back so we can see all of you";
const TOO_CLOSE_MIN_KEYPOINTS = 4;
const TOO_CLOSE_HINT_AFTER_MS = 1000;
let tooFewKeypointsSince = null;

// Cap on decoded images kept in memory (evicted least-recently-used).
// Uncapped, a kiosk running all day grows without bound and crashes the tab.
const IMAGE_CACHE_MAX = 40;
let imageCacheOrder = [];  // filenames, most recently used last
// Matching runs several times a second, far faster than an image loads, so
// track requests in flight (to avoid re-requesting the same file every tick)
// and recent failures (so a missing file isn't retried forever or re-picked).
let imageLoading = {};     // filename -> true while its request is in flight
let imageFailedAt = {};    // filename -> millis() when its load last failed
const IMAGE_RETRY_MS = 60000;  // leave a failed image alone for this long

// If loading hasn't finished after this long, say so instead of showing
// "Loading..." indefinitely (the model is ~9MB, so allow for slow connections)
const LOAD_WATCHDOG_MS = 45000;
let loadWatchdog = null;

// Maps each keypoint to its horizontal mirror (left/right swapped) in the
// MoveNet keypoint order. Used to also match against mirrored reference
// poses: a visitor raising their left arm should match artworks with either
// arm raised — this effectively doubles the pose coverage of the dataset.
const KEYPOINT_FLIP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15];

// Replace the loading placeholder's text (it is removed once setup() runs)
function showLoadMessage(message) {
  let loadingMsg = document.getElementById('loading-message');
  if (loadingMsg) loadingMsg.textContent = message;
}

// Debug overlay: add ?debug=1 to the page URL to show, live, how the
// "has the pose changed?" check is behaving — see drawDebugOverlay().
// Low-confidence keypoints are also drawn (orange rings) so you can see
// where the model is placing joints it can't actually see.
const DEBUG = new URLSearchParams(window.location.search).get('debug') === '1';
const DEBUG_WINDOW_MS = 5000;  // "low" figures cover this much recent history
let debugStats = null;         // latest figures, refreshed on every matching pass
let debugHistory = [];         // recent {time, all, masked} similarity samples
let debugHistoryFor = null;    // the matchPoseVector debugHistory was measured against
// Debug recording: every matching pass is logged in memory. Press a number
// key (1-9) to label what you are testing from then on, and D to download
// the whole log as a CSV file.
let debugLog = [];
let debugLabel = "";
const DEBUG_LOG_MAX = 50000;   // rows (about 3.5 hours at 4 per second)

function preload() {
  loadWatchdog = setTimeout(function() {
    showLoadMessage("This is taking longer than expected. Check your connection and reload the page.");
  }, LOAD_WATCHDOG_MS);

  // Load the bodyPose model from the locally hosted copy in libs/ so the
  // exhibit runs fully offline — without modelUrl, ml5 would download the
  // model weights from Google's servers on every page load
  bodyPose = ml5.bodyPose("MoveNet", {
    modelUrl: "libs/movenet-multipose-lightning-1/model.json"
  });

  // Load the pre-computed pose data for all reference images
  referencePoseData = loadJSON("input_image_metadata.json", function() {}, function() {
    clearTimeout(loadWatchdog);
    showLoadMessage("Could not load the artwork collection. Reload the page to try again.");
  });
}

function setup() {
  // Remove loading placeholder
  clearTimeout(loadWatchdog);
  let loadingMsg = document.getElementById('loading-message');
  if (loadingMsg) loadingMsg.remove();

  // Double width: webcam on left, matching image on right
  let canvas = createCanvas(800, 593);
  canvas.parent('canvas-container');

  // Accessibility: add aria-label to canvas
  canvas.elt.setAttribute('aria-label', 'Pose matching experience: your webcam feed on the left detects your pose, matched artwork appears on the right. ' +
    (KIOSK_MODE ? 'Scan the QR code to learn more.' : 'Click the artwork or scan the QR code to learn more.'));

  if (KIOSK_MODE) {
    // The "kiosk" class hides the About link (see index.html). Every other
    // link on the page (logo, footer) loses its destination and becomes
    // plain text. setup() runs after branding.js has wrapped the logo.
    document.body.classList.add('kiosk');
    for (let link of document.querySelectorAll('a')) {
      link.removeAttribute('href');
      link.removeAttribute('target');
    }
  }

  // Get the skeleton connection information
  connections = bodyPose.getSkeleton();

  startCamera();

  // Initialize QR code (empty initially). Generated larger than it is shown
  // so it stays sharp when the canvas is scaled up; CSS sets its on-screen size
  qrCode = new QRCode(document.getElementById("qrcode"), {
    width: 240,
    height: 240,
    colorDark: "#000000",
    colorLight: "#ffffff"
  });

}

// Set up the webcam. p5's createCapture logs a failed camera request to the
// console and nothing else, so first ask for the camera directly to learn
// whether (and why) it is unavailable. That probe stream is released before
// createCapture opens its own — leaving it open would hold the camera twice.
function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCameraError("Camera unavailable",
      "This page can only use the camera\nover https or on localhost");
    return;
  }
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(probeStream) {
      probeStream.getTracks().forEach(track => track.stop());
      video = createCapture(VIDEO, function() {
        cameraReady = true;
        // Start continuous pose detection on webcam
        bodyPose.detectStart(video, gotWebcamPoses);
      });
      video.size(640, 480);
      video.hide();
    })
    .catch(function(err) {
      console.log('Camera unavailable:', err.name, err.message);
      if (err.name === 'NotAllowedError' || err.name === 'SecurityError') {
        showCameraError("Camera access denied",
          "Please allow camera access\nto use Pose Match");
      } else if (err.name === 'NotFoundError' || err.name === 'OverconstrainedError') {
        showCameraError("No camera found",
          "Connect a camera and reload the page");
      } else if (err.name === 'NotReadableError' || err.name === 'AbortError') {
        showCameraError("Camera unavailable",
          "Another app may be using the camera.\nClose it and reload the page");
      } else {
        showCameraError("Camera unavailable", "Reload the page to try again");
      }
    });
}

function showCameraError(title, detail) {
  cameraError = true;
  cameraErrorTitle = title;
  cameraErrorDetail = detail;
}

function draw() {
  // Only update the match at the configured interval (not every frame)
  let currentTime = millis();
  if (currentTime - lastMatchTime >= MATCH_UPDATE_INTERVAL_MS) {
    findBestMatch();
    lastMatchTime = currentTime;
  }

  updateAttractMode(currentTime);

  // Draw webcam on the left side (or error message if camera denied)
  if (cameraError) {
    // Show error message
    fill(40);
    noStroke();
    rect(0, 0, 400, 593);
    fill(255);
    textAlign(CENTER, CENTER);
    textSize(18);
    text(cameraErrorTitle, 200, 270);
    textSize(14);
    fill(180);
    text(cameraErrorDetail, 200, 320);
  } else if (video) {
    // Crop-to-fit: scale webcam to fill area, clipping overflow (like object-fit: cover)
    let vidRatio = video.width / video.height;
    let areaW = 400;
    let areaH = 593;
    let areaRatio = areaW / areaH;
    let sx, sy, sw, sh;
    if (vidRatio > areaRatio) {
      // Webcam is wider than area — crop sides
      sh = video.height;
      sw = video.height * areaRatio;
      sx = (video.width - sw) / 2;
      sy = 0;
    } else {
      // Webcam is taller than area — crop top/bottom
      sw = video.width;
      sh = video.width / areaRatio;
      sx = 0;
      sy = (video.height - sh) / 2;
    }
    // Mirror the webcam horizontally — people expect to see themselves as
    // in a mirror; an unmirrored feed reads as "it's not following me"
    push();
    translate(areaW, 0);
    scale(-1, 1);
    image(video, 0, 0, areaW, areaH, sx, sy, sw, sh);
    pop();
    // Store crop params for skeleton mapping
    cropSx = sx; cropSy = sy; cropSw = sw; cropSh = sh;
    cropAreaW = areaW; cropAreaH = areaH;
  }

  // Draw skeleton for the person being matched (mirrored to match the video)
  if (primaryPose) {
    let pose = primaryPose;
    let mapX = x => cropAreaW - (x - cropSx) * (cropAreaW / cropSw);
    let mapY = y => (y - cropSy) * (cropAreaH / cropSh);

    // Draw the skeleton connections
    for (let j = 0; j < connections.length; j++) {
      let pointAIndex = connections[j][0];
      let pointBIndex = connections[j][1];
      let pointA = pose.keypoints[pointAIndex];
      let pointB = pose.keypoints[pointBIndex];
      // Only draw a line if both points are confident enough
      if (pointA.confidence > KEYPOINT_CONFIDENCE_MIN && pointB.confidence > KEYPOINT_CONFIDENCE_MIN) {
        stroke(255, 0, 0);
        strokeWeight(2);
        line(mapX(pointA.x), mapY(pointA.y), mapX(pointB.x), mapY(pointB.y));
      }
    }

    // Draw all the tracked landmark points
    for (let j = 0; j < pose.keypoints.length; j++) {
      let keypoint = pose.keypoints[j];
      if (keypoint.confidence > KEYPOINT_CONFIDENCE_MIN) {
        fill(0, 255, 0);
        noStroke();
        circle(mapX(keypoint.x), mapY(keypoint.y), 10);
      } else if (DEBUG) {
        // Where the model guesses a joint it can't see
        noFill();
        stroke(255, 160, 0);
        strokeWeight(2);
        circle(mapX(keypoint.x), mapY(keypoint.y), 10);
      }
    }
  }

  drawWebcamPrompt(currentTime);

  // Draw the best matching image on the right side (preserve aspect ratio)
  if (bestMatchImg) {
    let areaW = 400;
    let areaH = 593;
    let imgRatio = bestMatchImg.width / bestMatchImg.height;
    let areaRatio = areaW / areaH;
    let drawW, drawH;
    if (imgRatio > areaRatio) {
      // Image is wider than area — fit to width
      drawW = areaW;
      drawH = areaW / imgRatio;
    } else {
      // Image is taller than area — fit to height
      drawH = areaH;
      drawW = areaH * imgRatio;
    }
    let drawX = 400 + (areaW - drawW) / 2;
    let drawY = (areaH - drawH) / 2;
    // Black background behind image to fill letterbox area
    fill(0);
    noStroke();
    rect(400, 0, areaW, areaH);
    image(bestMatchImg, drawX, drawY, drawW, drawH);
  }

  // Draw label for the image actually on screen (not one still loading)
  if (displayedMatchData) {
    drawMatchLabel(displayedMatchData);
  }

  if (DEBUG) {
    drawDebugOverlay();
  }
}

// Record the figures the debug overlay shows. Called on every matching pass
// with the visitor's processed pose (null when too few keypoints are
// confident to attempt a match).
function updateDebugStats(webcamProcessed) {
  let now = millis();
  debugStats = {
    time: now,
    confident: primaryPose
      ? primaryPose.keypoints.filter(kp => kp.confidence > KEYPOINT_CONFIDENCE_MIN).length
      : 0,
    usable: webcamProcessed !== null,
    all: null,
    masked: null,
    matchScore: null,       // similarity of the image on screen to the pose now
    bestScore: null,        // best similarity available for the pose now
    aboveFloor: null,       // explore: images within the floor of the best
    freshAboveFloor: null,  // ...of which not shown recently or on screen
    pick: ""                // explore: fresh / recycled / hold
  };
  if (debugHistoryFor !== matchPoseVector) {
    debugHistory = [];  // a new match was chosen — the comparison pose changed
    debugHistoryFor = matchPoseVector;
  }
  if (webcamProcessed !== null && matchPoseVector !== null) {
    measureDebugSimilarity(webcamProcessed, now);
  }
}

// Append the finished matching pass to the debug recording
function logDebugRow() {
  if (debugLog.length >= DEBUG_LOG_MAX) {
    return;
  }
  let number = (value, places) => (value === null || !isFinite(value)) ? "" : value.toFixed(places);
  debugLog.push([
    Math.round(debugStats.time), debugLabel, debugStats.confident, debugStats.usable ? 1 : 0,
    number(debugStats.all, 4), number(debugStats.masked, 4),
    Math.round(millis() - matchStartTime), exploreMode ? "explore" : "best",
    bestMatchData ? bestMatchData.object_id : "",
    number(debugStats.matchScore, 4), number(debugStats.bestScore, 4),
    number(debugStats.aboveFloor, 0), number(debugStats.freshAboveFloor, 0), debugStats.pick
  ].concat(primaryPose ? primaryPose.keypoints.map(kp => kp.confidence.toFixed(2)) : []));
}

function measureDebugSimilarity(webcamProcessed, now) {
  // What the matcher uses: only joints confident both now and at match time
  debugStats.masked = similarityToMatchPose(webcamProcessed);
  // For comparison: every joint, including ones the model can't see
  debugStats.all = cosineSimilarity(webcamProcessed.l2Vector, matchPoseVector);
  debugHistory.push({ time: now, all: debugStats.all, masked: debugStats.masked });
  while (debugHistory.length > 0 && now - debugHistory[0].time > DEBUG_WINDOW_MS) {
    debugHistory.shift();
  }
}

// Draw the debug panel over the bottom of the webcam half
function drawDebugOverlay() {
  let lines = ["DEBUG  keys: 1-9 label test, D download log",
    "recording: " + debugLog.length + " samples, label: " + (debugLabel || "(none)")];
  if (!primaryPose || !debugStats || millis() - debugStats.time > 1000) {
    lines.push("no person detected" + (attractMode ? "  (attract mode)" : ""));
  } else {
    lines.push("confident keypoints: " + debugStats.confident + " of 17" +
      (debugStats.usable ? "" : "  - too few to match (need " + MIN_CONFIDENT_KEYPOINTS + ")"));
    if (debugStats.all === null) {
      lines.push("no image chosen yet for this pose");
    } else {
      let lowAll = Math.min(...debugHistory.map(sample => sample.all));
      let lowMasked = Math.min(...debugHistory.map(sample => sample.masked));
      let verdict = value => value < MOVEMENT_THRESHOLD ? "CHANGED" : "same";
      lines.push("similarity to your pose when this image was chosen");
      lines.push("(below " + MOVEMENT_THRESHOLD + " = pose counts as changed)");
      lines.push("                    now     low " + (DEBUG_WINDOW_MS / 1000) + "s   pose");
      lines.push("visible joints(used)" + debugStats.masked.toFixed(3) + "   " + lowMasked.toFixed(3) + "    " + verdict(debugStats.masked));
      lines.push("all joints (old)    " + debugStats.all.toFixed(3) + "   " + lowAll.toFixed(3) + "    " + verdict(debugStats.all));
      let minDisplay = debugStats.masked < MOVEMENT_THRESHOLD ? EXPLORE_MIN_DISPLAY_MOVING_MS : EXPLORE_MIN_DISPLAY_STILL_MS;
      lines.push("image shown " + ((millis() - matchStartTime) / 1000).toFixed(1) + "s, can change after " +
        (minDisplay / 1000).toFixed(1) + "s  (" + (exploreMode ? "explore" : "best match") + ")");
    }
    if (debugStats.bestScore !== null && isFinite(debugStats.bestScore)) {
      lines.push("match score: this image " +
        (debugStats.matchScore === null ? "-" : debugStats.matchScore.toFixed(3)) +
        ", best available " + debugStats.bestScore.toFixed(3));
      if (debugStats.aboveFloor !== null) {
        lines.push("within " + EXPLORE_SIMILARITY_FLOOR + " of best: " + debugStats.aboveFloor +
          " images, " + debugStats.freshAboveFloor + " unseen  (next: " + debugStats.pick + ")");
      }
    }
  }

  let lineHeight = 14;
  let boxHeight = lines.length * lineHeight + 12;
  push();
  noStroke();
  fill(0, 0, 0, 190);
  rect(0, 593 - boxHeight, 400, boxHeight);
  fill(255);
  textFont('monospace');
  textSize(11);
  textAlign(LEFT, TOP);
  for (let i = 0; i < lines.length; i++) {
    text(lines[i], 8, 593 - boxHeight + 6 + i * lineHeight);
  }
  pop();
}

// Label layout (canvas units). The label sits on a dark translucent band so
// it stays readable over light artworks.
const LABEL_MAX_WIDTH = 376;       // text width within the 400px artwork half
const LABEL_TITLE_SIZE = 14;
const LABEL_TITLE_LEADING = 18;
const LABEL_TITLE_MAX_LINES = 2;   // longer titles end in an ellipsis
const LABEL_ARTIST_SIZE = 12;
const LABEL_PADDING = 8;
let labelLayout = null;            // wrapped lines for the match on screen
let labelLayoutFor = null;         // the match labelLayout was built for

// Draw the title and artist of the matched artwork across the top of the
// artwork half. The artist line is left out when the dataset has none.
function drawMatchLabel(match) {
  if (labelLayoutFor !== match) {
    textSize(LABEL_TITLE_SIZE);
    let titleLines = wrapLabelText(match.metadata.title || '', LABEL_MAX_WIDTH, LABEL_TITLE_MAX_LINES);
    textSize(LABEL_ARTIST_SIZE);
    let artistLines = wrapLabelText(match.metadata.artist || '', LABEL_MAX_WIDTH, 1);
    labelLayout = { titleLines: titleLines, artist: artistLines[0] || '' };
    labelLayoutFor = match;
  }
  let titleLines = labelLayout.titleLines;
  let artist = labelLayout.artist;
  if (titleLines.length === 0 && !artist) {
    return;
  }

  let titleHeight = titleLines.length * LABEL_TITLE_LEADING;
  let artistHeight = artist ? LABEL_ARTIST_SIZE + 6 : 0;
  let bandHeight = LABEL_PADDING + titleHeight + artistHeight + LABEL_PADDING;

  noStroke();
  fill(0, 0, 0, 185);
  rect(400, 0, 400, bandHeight);

  textAlign(CENTER, TOP);
  fill(255);
  textSize(LABEL_TITLE_SIZE);
  for (let i = 0; i < titleLines.length; i++) {
    text(titleLines[i], 600, LABEL_PADDING + i * LABEL_TITLE_LEADING);
  }
  if (artist) {
    fill(210);
    textSize(LABEL_ARTIST_SIZE);
    text(artist, 600, LABEL_PADDING + titleHeight + 2);
  }
}

// Word-wrap a string to at most maxLines lines of maxWidth, measured at the
// current textSize. Text that doesn't fit ends in an ellipsis.
function wrapLabelText(str, maxWidth, maxLines) {
  let words = String(str).trim().split(/\s+/).filter(w => w.length > 0);
  let lines = [];
  let line = '';
  let i = 0;
  while (i < words.length) {
    let candidate = line ? line + ' ' + words[i] : words[i];
    if (textWidth(candidate) <= maxWidth || !line) {
      // fits — or it is a single word too long for any line, which is kept
      // whole here and cut to fit below
      line = candidate;
      i++;
    } else if (lines.length === maxLines - 1) {
      break;  // out of lines with words left over
    } else {
      lines.push(line);
      line = '';
    }
  }
  if (line) {
    lines.push(line);
  }
  let truncated = i < words.length;
  for (let n = 0; n < lines.length; n++) {
    let isLast = n === lines.length - 1;
    let needsEllipsis = isLast && truncated;
    if (!needsEllipsis && textWidth(lines[n]) <= maxWidth) {
      continue;
    }
    let cut = lines[n];
    while (cut.length > 1 && textWidth(cut + '\u2026') > maxWidth) {
      cut = cut.slice(0, -1).trimEnd();
    }
    lines[n] = cut + '\u2026';
  }
  return lines;
}

// Callback for continuous webcam pose detection
function gotWebcamPoses(results) {
  webcamPoses = results;
  primaryPose = selectPrimaryPose(results);
}

// Enter attract mode when nobody has been matchable for a while, and run its
// slideshow. (Leaving it happens in runMatchingPass, when a matchable pose
// appears.)
function updateAttractMode(now) {
  if (!attractMode && lastMatchableTime !== null && now - lastMatchableTime > ATTRACT_AFTER_MS) {
    attractMode = true;
    attractNextSlideTime = now;
    // The next visitor starts fresh
    recentMatches = [];
    matchPoseVector = null;
    // Tell screen reader users once, rather than announcing every slide
    if (announceTimeout) {
      clearTimeout(announceTimeout);
    }
    let announcement = document.getElementById('match-announcement');
    if (announcement) {
      announcement.textContent = ATTRACT_PROMPT;
    }
  }
  if (attractMode && now >= attractNextSlideTime && referencePoseData !== null) {
    attractNextSlideTime = now + ATTRACT_SLIDE_MS;
    showAttractSlide();
  }
}

// Put a random artwork on the artwork half
function showAttractSlide() {
  if (referenceList === null) {
    buildReferenceList();
  }
  let current = bestMatchData ? bestMatchData.filename : null;
  let choices = referenceList.filter(entry =>
    entry.data.filename !== current && imageFailedAt[entry.data.filename] === undefined);
  if (choices.length === 0) {
    return;
  }
  bestMatchData = choices[Math.floor(Math.random() * choices.length)].data;
  matchStartTime = millis();
  showCurrentMatch();
}

// Draw a prompt across the webcam half: the too-close hint when someone is
// in view but can't be matched, otherwise the attract prompt in attract mode.
function drawWebcamPrompt(now) {
  let confident = primaryPose
    ? primaryPose.keypoints.filter(kp => kp.confidence > KEYPOINT_CONFIDENCE_MIN).length
    : 0;
  if (confident >= TOO_CLOSE_MIN_KEYPOINTS && confident < MIN_CONFIDENT_KEYPOINTS) {
    if (tooFewKeypointsSince === null) {
      tooFewKeypointsSince = now;
    }
  } else {
    tooFewKeypointsSince = null;
  }

  let message = null;
  if (tooFewKeypointsSince !== null && now - tooFewKeypointsSince >= TOO_CLOSE_HINT_AFTER_MS) {
    message = TOO_CLOSE_HINT;
  } else if (attractMode) {
    message = ATTRACT_PROMPT;
  }
  if (message === null || cameraError) {
    return;
  }

  push();
  textStyle(BOLD);
  textSize(22);
  let lines = wrapLabelText(message, 368, 3);
  let leading = 28;
  let bandHeight = lines.length * leading + 24;
  let bandTop = (593 - bandHeight) / 2;
  noStroke();
  fill(0, 0, 0, 185);
  rect(0, bandTop, 400, bandHeight);
  fill(255);
  textAlign(CENTER, TOP);
  for (let i = 0; i < lines.length; i++) {
    text(lines[i], 200, bandTop + 12 + i * leading + 3);
  }
  pop();
}

// With multiple people in frame, match the one taking up the most space
// (closest to the camera) instead of whichever the model listed first —
// otherwise the match can flip between people from frame to frame.
function selectPrimaryPose(poses) {
  let best = null;
  let bestArea = -1;
  for (let pose of poses) {
    if (!pose || !pose.keypoints) {
      continue;
    }
    let box = computeBoundingBox(pose.keypoints, KEYPOINT_CONFIDENCE_MIN);
    if (box.xMin === Infinity) {
      continue;  // no confident keypoints at all
    }
    let area = (box.xMax - box.xMin) * (box.yMax - box.yMin);
    if (area > bestArea) {
      bestArea = area;
      best = pose;
    }
  }
  return best;
}

// Build the pre-processed reference list once: unwrap the loadJSON object,
// and precompute a horizontally mirrored copy of every pose vector (and its
// keypoint confidences) so each artwork can match either chirality.
function buildReferenceList() {
  let poseArray = Array.isArray(referencePoseData) ? referencePoseData : Object.values(referencePoseData);
  referenceList = [];
  for (let reference of poseArray) {
    if (!reference.l2_vector) {
      continue;
    }
    let confidences = reference.keypoint_confidences || null;
    referenceList.push({
      data: reference,
      vector: reference.l2_vector,
      confidences: confidences,
      flippedVector: flipPoseVector(reference.l2_vector),
      flippedConfidences: confidences ? KEYPOINT_FLIP_MAP.map(i => confidences[i]) : null
    });
  }
}

// Mirror a flattened pose vector horizontally: negate x and swap left/right
// keypoints. The vectors are centered on the bounding box, so negating x
// mirrors in place; L2 magnitude is unchanged.
function flipPoseVector(vector) {
  let flipped = new Array(vector.length);
  for (let k = 0; k < KEYPOINT_FLIP_MAP.length; k++) {
    let src = KEYPOINT_FLIP_MAP[k] * 2;
    flipped[k * 2] = -vector[src];
    flipped[k * 2 + 1] = vector[src + 1];
  }
  return flipped;
}

// Run one matching pass (and, in debug mode, record it)
function findBestMatch() {
  let statsBefore = debugStats;
  runMatchingPass();
  if (DEBUG && debugStats !== statsBefore) {
    logDebugRow();
  }
}

// Find the reference pose that best matches the webcam pose
function runMatchingPass() {
  // Need webcam pose and reference poses to compare
  if (primaryPose === null || referencePoseData === null) {
    return;
  }

  // Process the webcam pose
  let webcamProcessed = processPose(primaryPose);
  if (DEBUG) {
    updateDebugStats(webcamProcessed);
  }
  if (webcamProcessed === null) {
    return;
  }

  lastMatchableTime = millis();
  if (attractMode) {
    // Someone is here. Forget the slideshow artwork as the "current match"
    // so their first match appears at once, without the minimum display
    // time, and the slide isn't counted as an image they were shown. (It
    // stays on screen until that first match has loaded.)
    attractMode = false;
    bestMatchData = null;
  }

  if (referenceList === null) {
    buildReferenceList();
  }

  let bestSimilarity = -Infinity;
  let bestData = null;
  let onScreenSimilarity = null;  // how well the image on screen fits the pose now
  let candidates = [];  // explore mode: every image except the one on screen

  // Loop through all reference poses and find the best match
  for (let i = 0; i < referenceList.length; i++) {
    let entry = referenceList[i];

    // Skip artworks whose image recently failed to load
    let failedAt = imageFailedAt[entry.data.filename];
    if (failedAt !== undefined) {
      if (millis() - failedAt < IMAGE_RETRY_MS) {
        continue;
      }
      delete imageFailedAt[entry.data.filename];
    }

    // Compare against the artwork's pose and its mirror image, over only
    // the keypoints both sides are confident about
    let similarity = Math.max(
      maskedCosineSimilarity(webcamProcessed.l2Vector, entry.vector,
                             webcamProcessed.keypointMask, entry.confidences),
      maskedCosineSimilarity(webcamProcessed.l2Vector, entry.flippedVector,
                             webcamProcessed.keypointMask, entry.flippedConfidences)
    );

    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestData = entry.data;
    }

    // In explore mode, recently shown images are ineligible — only fresh
    // images compete. This is what guarantees variety: for distinctive poses
    // the few close matches would otherwise win every rotation. The image
    // on screen is ineligible too: it only joins recentMatches when it is
    // replaced, so if it could compete, a pose with one clear best match
    // would hold that image indefinitely.
    let isOnScreen = bestMatchData !== null && entry.data.filename === bestMatchData.filename;
    if (isOnScreen) {
      onScreenSimilarity = similarity;
    } else if (exploreMode) {
      candidates.push({
        similarity: similarity,
        reference: entry.data,
        recentIndex: recentMatches.indexOf(entry.data.filename)  // -1 = not shown recently
      });
    }
  }

  if (exploreMode) {
    // Only images within EXPLORE_SIMILARITY_FLOOR of the best match compete
    let aboveFloor = candidates.filter(c => c.similarity >= bestSimilarity - EXPLORE_SIMILARITY_FLOOR);
    let fresh = aboveFloor.filter(c => c.recentIndex === -1);
    let pick = "hold";
    if (fresh.length > 0) {
      // Pick randomly among the near-tie fresh candidates rather than always
      // taking the single best. The min-display gate below still controls
      // when the displayed image is actually allowed to rotate.
      let bestFresh = Math.max(...fresh.map(c => c.similarity));
      let band = fresh.filter(c => c.similarity >= bestFresh - MATCH_SAMPLE_EPSILON);
      band.sort((a, b) => b.similarity - a.similarity);
      if (band.length > MATCH_SAMPLE_TOP_K) {
        band.length = MATCH_SAMPLE_TOP_K;
      }
      bestData = band[Math.floor(Math.random() * band.length)].reference;
      pick = "fresh";
    } else if (aboveFloor.length > 0) {
      // Every image above the floor has been shown: go back to the one
      // shown longest ago rather than dropping to a poor match
      let oldest = aboveFloor.reduce((a, b) => (b.recentIndex < a.recentIndex ? b : a));
      bestData = oldest.reference;
      pick = "recycled";
    } else if (bestMatchData !== null) {
      // Nothing but the image on screen clears the floor (or the pose was
      // degenerate and nothing could be scored) — keep showing it
      bestData = bestMatchData;
    }
    if (DEBUG) {
      let onScreenAbove = onScreenSimilarity !== null &&
        onScreenSimilarity >= bestSimilarity - EXPLORE_SIMILARITY_FLOOR ? 1 : 0;
      debugStats.aboveFloor = aboveFloor.length + onScreenAbove;
      debugStats.freshAboveFloor = fresh.length;
      debugStats.pick = pick;
    }
  }
  if (DEBUG) {
    debugStats.matchScore = onScreenSimilarity;
    debugStats.bestScore = bestSimilarity;
  }

  // If we found a match, update the display
  if (bestData !== null) {
    let matchChanged = bestData.filename !== (bestMatchData && bestMatchData.filename);

    if (matchChanged) {
      // A different image wants to win — only allow the switch if the current image
      // has been shown for the minimum display time. This prevents rapid cycling when
      // the pose naturally fluctuates between a few similar images. The minimum is
      // short once the visitor has left the pose that produced the current image,
      // so a newly struck pose gets its match promptly, in either mode.
      let poseChanged = matchPoseVector !== null &&
        similarityToMatchPose(webcamProcessed) < MOVEMENT_THRESHOLD;
      let minDisplay = poseChanged ? EXPLORE_MIN_DISPLAY_MOVING_MS : EXPLORE_MIN_DISPLAY_STILL_MS;

      if (bestMatchData === null || millis() - matchStartTime >= minDisplay) {
        // Allowed to switch. An image coming back from the recent list leaves
        // it while on screen (it rejoins at the end when it is replaced)
        let incomingIndex = recentMatches.indexOf(bestData.filename);
        if (incomingIndex !== -1) {
          recentMatches.splice(incomingIndex, 1);
        }
        // Penalize the outgoing image now that we're done with it
        if (exploreMode && bestMatchData &&
            recentMatches.indexOf(bestMatchData.filename) === -1) {
          recentMatches.push(bestMatchData.filename);
          // Cap history below the dataset size so fresh images always remain
          // eligible — a dataset smaller than EXPLORE_HISTORY_SIZE would
          // otherwise permanently exhaust and freeze explore mode
          let historyCap = Math.min(EXPLORE_HISTORY_SIZE, Math.floor(referenceList.length / 2));
          while (recentMatches.length > historyCap) {
            recentMatches.shift();
          }
        }
        matchStartTime = millis();
        bestMatchData = bestData;
        matchPoseVector = webcamProcessed.l2Vector;
        matchPoseMask = webcamProcessed.keypointMask;
      }
      // else: minimum display time not yet elapsed — keep showing the current match
    }

    showCurrentMatch();
  }
}

// Show bestMatchData on the artwork half, loading its image if needed.
function showCurrentMatch() {
  if (bestMatchData === null) {
    return;
  }
  // Swap in the image — and only once it's actually visible, update the
  // label/QR/announcement via displayedMatchData, so they never describe
  // an image that hasn't loaded yet
  let filename = bestMatchData.filename;
  if (imageCache[filename]) {
    bestMatchImg = imageCache[filename];
    touchImageCache(filename);
    if (displayedMatchData !== bestMatchData) {
      displayedMatchData = bestMatchData;
      updateQRCode(displayedMatchData.object_id);
    }
  } else if (!imageLoading[filename]) {
    imageLoading[filename] = true;
    loadImage("input_images/" + filename, function(img) {
      delete imageLoading[filename];
      imageCache[filename] = img;
      touchImageCache(filename);
      // Only swap in if this is still the match we're switching to
      if (bestMatchData && bestMatchData.filename === filename) {
        bestMatchImg = img;
        displayedMatchData = bestMatchData;
        updateQRCode(displayedMatchData.object_id);
      }
    }, function() {
      // Missing or unreadable image: stop waiting on it. Fall back to the
      // image still on screen so the next matching pass can pick another
      // artwork right away (this one is skipped for IMAGE_RETRY_MS).
      delete imageLoading[filename];
      imageFailedAt[filename] = millis();
      console.log('Could not load image:', filename);
      if (bestMatchData && bestMatchData.filename === filename) {
        bestMatchData = displayedMatchData;
        matchStartTime = 0;
        // in attract mode, try another slide shortly (not instantly — if every
        // image is failing, that would fire off requests as fast as they fail)
        attractNextSlideTime = millis() + 1000;
      }
    });
  }
}

// Similarity between the visitor's current pose and their pose when the
// on-screen image was chosen, over the keypoints confident in both.
// Only call when matchPoseVector is set.
function similarityToMatchPose(webcamProcessed) {
  return maskedCosineSimilarity(
    webcamProcessed.l2Vector, matchPoseVector,
    webcamProcessed.keypointMask, matchPoseMask.map(confident => confident ? 1 : 0));
}

// Mark an image as recently used and evict the least-recently-used images
// beyond the cache cap. Evicted p5.Images are garbage-collected once nothing
// references them (the on-screen image stays alive via bestMatchImg).
function touchImageCache(filename) {
  let idx = imageCacheOrder.indexOf(filename);
  if (idx !== -1) {
    imageCacheOrder.splice(idx, 1);
  }
  imageCacheOrder.push(filename);
  while (imageCacheOrder.length > IMAGE_CACHE_MAX) {
    let evicted = imageCacheOrder.shift();
    delete imageCache[evicted];
  }
}

// Compute bounding box from keypoints with confidence threshold
function computeBoundingBox(keypoints, confidenceThreshold = 0.1) {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;

  for (let keypoint of keypoints) {
    if (keypoint.confidence > confidenceThreshold) {
      xMin = Math.min(xMin, keypoint.x);
      xMax = Math.max(xMax, keypoint.x);
      yMin = Math.min(yMin, keypoint.y);
      yMax = Math.max(yMax, keypoint.y);
    }
  }

  return { xMin, xMax, yMin, yMax };
}

// Normalize keypoints: center around bounding box and scale to unit square
function normalizeKeypoints(keypoints, confidenceThreshold = 0.1) {
  let box = computeBoundingBox(keypoints, confidenceThreshold);

  // Compute center and size of bounding box
  let centerX = (box.xMin + box.xMax) / 2;
  let centerY = (box.yMin + box.yMax) / 2;
  let boxWidth = box.xMax - box.xMin;
  let boxHeight = box.yMax - box.yMin;

  // Use the larger dimension to maintain aspect ratio
  let scale = Math.max(boxWidth, boxHeight);

  // Avoid division by zero
  if (scale === 0) {
    scale = 1;
  }

  // Normalize each keypoint: center and scale to [-0.5, 0.5] range
  let normalized = [];
  for (let keypoint of keypoints) {
    normalized.push({
      name: keypoint.name,
      x: (keypoint.x - centerX) / scale,
      y: (keypoint.y - centerY) / scale,
      confidence: keypoint.confidence
    });
  }

  return normalized;
}

// Flatten keypoints to a vector [x1, y1, x2, y2, ...] for L2 normalization
function keypointsToVector(keypoints) {
  let vector = [];
  for (let keypoint of keypoints) {
    vector.push(keypoint.x);
    vector.push(keypoint.y);
  }
  return vector;
}

// L2 normalize a vector (divide by magnitude to create unit vector)
function l2Normalize(vector) {
  let sumOfSquares = 0;
  for (let val of vector) {
    sumOfSquares += val * val;
  }
  let magnitude = Math.sqrt(sumOfSquares);

  // Avoid division by zero
  if (magnitude === 0) {
    return vector;
  }

  return vector.map(val => val / magnitude);
}

// Process a pose: normalize keypoints and apply L2 normalization.
// Returns null when too few keypoints are confident — a vector built mostly
// from occluded joints is noise and would produce arbitrary matches.
function processPose(pose) {
  if (!pose || !pose.keypoints) {
    return null;
  }

  let keypointMask = pose.keypoints.map(kp => kp.confidence > KEYPOINT_CONFIDENCE_MIN);
  let confidentCount = keypointMask.filter(Boolean).length;
  if (confidentCount < MIN_CONFIDENT_KEYPOINTS) {
    return null;
  }

  let normalizedKeypoints = normalizeKeypoints(pose.keypoints);
  let vector = keypointsToVector(normalizedKeypoints);
  let l2Vector = l2Normalize(vector);

  return {
    normalizedKeypoints: normalizedKeypoints,
    l2Vector: l2Vector,
    keypointMask: keypointMask
  };
}

// Cosine similarity over only the keypoints both poses are confident about.
// The webcam side supplies a boolean mask; the reference side supplies its
// per-keypoint confidences when the dataset includes them (older datasets
// don't — then only the webcam mask applies). Both vectors are renormalized
// over the shared dimensions so partial poses compare fairly.
function maskedCosineSimilarity(vectorA, vectorB, keypointMaskA, refConfidences) {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let k = 0; k < keypointMaskA.length; k++) {
    if (!keypointMaskA[k]) {
      continue;
    }
    if (refConfidences && refConfidences[k] <= KEYPOINT_CONFIDENCE_MIN) {
      continue;
    }
    let d = k * 2;
    dot += vectorA[d] * vectorB[d] + vectorA[d + 1] * vectorB[d + 1];
    magA += vectorA[d] * vectorA[d] + vectorA[d + 1] * vectorA[d + 1];
    magB += vectorB[d] * vectorB[d] + vectorB[d + 1] * vectorB[d + 1];
  }
  if (magA === 0 || magB === 0) {
    return -1;  // no shared confident keypoints — treat as a non-match
  }
  return dot / Math.sqrt(magA * magB);
}

// Compute cosine similarity between two L2-normalized vectors
// Returns a value from -1 to 1 (1 = identical, 0 = unrelated)
function cosineSimilarity(vectorA, vectorB) {
  if (vectorA.length !== vectorB.length) {
    console.error("Vectors must have the same length");
    return 0;
  }

  let dotProduct = 0;
  for (let i = 0; i < vectorA.length; i++) {
    dotProduct += vectorA[i] * vectorB[i];
  }

  return dotProduct;
}

// Build the info page URL for an artwork. The id is URL-encoded because
// object_ids can contain spaces and other characters — phone scanners cut a
// link off at a space. baseUrl must end in "/"; omit it to link to wherever
// this page is served from.
function infoPageUrl(objectId, baseUrl) {
  let base = baseUrl || window.location.href.replace(/[?#].*$/, '').replace(/[^/]*$/, '');
  return base + 'info.html?id=' + encodeURIComponent(objectId);
}

// Update the QR code when the match changes
function updateQRCode(objectId) {
  if (qrCode && objectId !== currentQRObjectId) {
    currentQRObjectId = objectId;
    qrCode.clear();
    qrCode.makeCode(infoPageUrl(objectId, QR_BASE_URL));
    // The QR box stays hidden until there is a code to show
    document.getElementById("qrcode").style.display = "block";

    // Debounced screen reader announcement - only announce after match is stable for a period of time (avoids a barrage of announcements)
    if (announceTimeout) {
      clearTimeout(announceTimeout);
    }
    if (!attractMode) {
      announceTimeout = setTimeout(announceMatch, ANNOUNCE_DELAY_MS);
    }
  }
}

// Announce the current match to screen readers
function announceMatch() {
  let announcement = document.getElementById('match-announcement');
  if (announcement && displayedMatchData && !attractMode) {
    let title = displayedMatchData.metadata.title || 'Untitled';
    let artist = displayedMatchData.metadata.artist || 'Unknown artist';
    announcement.textContent = 'Matched artwork: ' + title + ' by ' + artist;
  }
}

// Debug recording keys (only with ?debug=1): 1-9 label, D download
function keyPressed() {
  if (!DEBUG) {
    return;
  }
  if (key >= '1' && key <= '9') {
    debugLabel = key;
  } else if (key === 'd' || key === 'D') {
    let header = ["ms", "label", "confident_keypoints", "usable", "similarity_all_joints",
      "similarity_hidden_masked", "ms_image_shown", "mode", "match_object_id",
      "match_score", "best_score", "above_floor", "unseen_above_floor", "next_pick"]
      .concat(KEYPOINT_FLIP_MAP.map((_, i) => "conf_" + i));
    let quote = value => '"' + String(value).replace(/"/g, '""') + '"';
    let csv = [header].concat(debugLog).map(row => row.map(quote).join(",")).join("\n");
    let link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    link.download = "pose-match-debug.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  }
}

// Click on matched image to open info page
function mousePressed() {
  // Only respond to left-clicks (allow right-click context menu)
  if (mouseButton !== LEFT || KIOSK_MODE) {
    return;
  }

  // Only respond to clicks on the right side (matched image area)
  if (mouseX > 400 && mouseX < 800 && mouseY > 0 && mouseY < 593) {
    if (displayedMatchData) {
      window.open(infoPageUrl(displayedMatchData.object_id), '_blank');
    }
  }
}
