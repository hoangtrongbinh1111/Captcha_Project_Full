const fs = require('fs')
const path = require('path')

exports.getDir = ({ dir }) => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    // Set directory owner to user running Node.js process
    const uid = process.getuid();
    const gid = process.getgid();
    fs.chownSync(dir, uid, gid);
    fs.chmodSync(dir, '777');
    return dir;
}

exports.removeDir = ({ dir }) => {
    fs.rmSync(dir, { recursive: true, force: true });
}

exports.createFile = ({ dir }) => {
    fs.open(dir, 'w', (err) => {
        if (err) throw err;
    });
}

exports.saveFileInfer = ({ filename, file }) => {
    fs.writeFile(filename, file, (err) => {
        if (err) throw err;
    });
}

