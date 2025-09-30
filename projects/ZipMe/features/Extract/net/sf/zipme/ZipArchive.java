

package net.sf.zipme;
class ZipArchive {

   public InputStream getInputStream(  ZipEntry entry) throws IOException {
    Hashtable entries=getEntries();
    String name=entry.getName();
    ZipEntry zipEntry=(ZipEntry)entries.get(name);
    if (zipEntry == null)     return null;
    ZipArchive_PartialInputStream inp=new ZipArchive_PartialInputStream(buf,off,len);
    inp.seek(off + zipEntry.offset);
    if (inp.readLeInt() != LOCSIG)     throw new ZipException("Wrong Local header signature: " + name);
    inp.skip(4);
    if (zipEntry.getMethod() != inp.readLeShort())     throw new ZipException("Compression method mismatch: " + name);
    inp.skip(16);
    int nameLen=inp.readLeShort();
    int extraLen=inp.readLeShort();
    inp.skip(nameLen + extraLen);
    inp.setLength((int)zipEntry.getCompressedSize());
    int method=zipEntry.getMethod();
switch (method) {
case ZipOutputStream.STORED:
      return inp;
case ZipOutputStream.DEFLATED:
    inp.addDummyByte();
  final Inflater inf=new Inflater(true);
final int sz=(int)entry.getSize();
return new ZipArchive_InflaterInputStream(inp,inf,sz);
default :
throw new ZipException("Unknown compression method " + method);
}
}
}
