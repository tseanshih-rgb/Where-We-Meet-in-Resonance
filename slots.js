
inlets = 1;
outlets = 2;

var MAXV = (jsarguments.length > 1) ? jsarguments[1] : 8;

//interval table, most consonant -> most dissonant
var INTERVALS = [0, 7, 4, 5, 9, 2, 11, 6];

var slotOf = {};
var freeSlots = [];
for (var i = MAXV; i >= 1; i--) freeSlots.push(i);   // pop() hands out 1 first

function anything() {
    var addr = messagename;                 //"/tracking/163/x"
    var args = arrayfromargs(arguments);
    var parts = addr.split("/");
    if (parts.length < 3 || parts[1] != "tracking") return;

    var seg = parts[2];

    if (seg == "enter"){
		assign(args[0]); return;
	}
    if (seg == "exit"){
		release(args[0]); return;
	}

    if (seg !== "" && !isNaN(Number(seg))){ //not空的&&number
        var id = parseInt(seg); //to int
        var param = parts.slice(3).join("/");   //x, speed, bbox/w...
        var slot = slotOf[id];
        if (slot == undefined) slot = assign(id);   //enter missed -> recover
        if (slot == -1) return;                     //all channels are busy -> ignore this person
        if (args.length > 0) {
            outlet(0, "target", slot);
            outlet(0, param, args[0]);
        }
        return;
    }

    //global message: /tracking/count, dispersion, nearest, present...
    var name = parts.slice(2).join("/");
    if (args.length > 0) outlet(1, name, args[0]); //get"dispersion 0.76"
}

function assign(id) {
    if (slotOf[id] != undefined) return slotOf[id];
    if (freeSlots.length == 0) return -1;
    var slot = freeSlots.pop();
    slotOf[id] = slot;
    outlet(0, "target", slot);
    outlet(0, "interval", INTERVALS[(slot - 1) % INTERVALS.length]);
    outlet(0, "on", 1);
    return slot;
}

function release(id) {
    var slot = slotOf[id];
    if (slot == undefined) return;
    outlet(0, "target", slot);
    outlet(0, "on", 0);
    delete slotOf[id];
    freeSlots.push(slot);
    freeSlots.sort(function (a, b) { return b - a; });  // keep low slots first
}

//message "reset": free every slot
function reset() {
    for (var id in slotOf) release(parseInt(id));
}
